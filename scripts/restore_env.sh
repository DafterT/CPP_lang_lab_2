#!/usr/bin/env bash
set -euo pipefail

# ---------- найти правильный state.env ----------
STATE_FILE="${STATE_FILE:-$HOME/.bench_env/state.env}"

# если под sudo и файл не найден — попробуем взять из дома исходного пользователя
if [[ ! -f "$STATE_FILE" && -n "${SUDO_USER:-}" ]]; then
  SUDO_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6 || true)"
  CANDIDATE="${SUDO_HOME:-}/.bench_env/state.env"
  [[ -f "$CANDIDATE" ]] && STATE_FILE="$CANDIDATE"
fi

[[ -f "$STATE_FILE" ]] || { echo "[restore][error] not found: $STATE_FILE"; exit 1; }

CPU_PATH="/sys/devices/system/cpu"
INTEL_TURBO="/sys/devices/system/cpu/intel_pstate/no_turbo"
GENERIC_BOOST="/sys/devices/system/cpu/cpufreq/boost"
ASLR_FILE="/proc/sys/kernel/randomize_va_space"
SMT_CTRL="/sys/devices/system/cpu/smt/control"
NMI_FILE="/proc/sys/kernel/nmi_watchdog"
THP_ENABLED="/sys/kernel/mm/transparent_hugepage/enabled"
THP_DEFRAG="/sys/kernel/mm/transparent_hugepage/defrag"

trim() { local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
active_from_thp_line() {
  local s; s="$(trim "$1")"
  if [[ "$s" =~ \[([^\]]+)\] ]]; then echo "${BASH_REMATCH[1]}"; else set -- $s; echo "${1:-}"; fi
}

# ---------- парсинг state.env ----------
declare -A KV CPU_GOV CPU_MIN CPU_MAX CPU_EPP IRQ_AFF
IRQ_LIST=""

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^\# ]] && continue
  [[ "$line" != *"="* ]] && continue
  key="$(trim "${line%%=*}")"
  val="$(trim "${line#*=}")"
  KV["$key"]="$val"

  case "$key" in
    CPU[0-9]*_GOV) idx=${key#CPU}; idx=${idx%_GOV}; CPU_GOV["$idx"]="$val" ;;
    CPU[0-9]*_MIN) idx=${key#CPU}; idx=${idx%_MIN}; CPU_MIN["$idx"]="$val" ;;
    CPU[0-9]*_MAX) idx=${key#CPU}; idx=${idx%_MAX}; CPU_MAX["$idx"]="$val" ;;
    CPU[0-9]*_EPP) idx=${key#CPU}; idx=${idx%_EPP}; CPU_EPP["$idx"]="$val" ;;
    IRQ_LIST) IRQ_LIST="$val" ;;
    IRQ[0-9]*_AFFINITY_LIST) irq="${key#IRQ}"; irq="${irq%_AFFINITY_LIST}"; IRQ_AFF["$irq"]="$val" ;;
  esac
done < "$STATE_FILE"

ASLR="${KV[ASLR]:-}"
INTEL_NO_TURBO="${KV[INTEL_NO_TURBO]:-}"
CPUFREQ_BOOST="${KV[CPUFREQ_BOOST]:-}"
SMT_VAL="${KV[SMT_CONTROL]:-}"
NMI_VAL="${KV[NMI_WATCHDOG]:-}"

# THP: извлечь активные режимы
THP_ENABLED_ACTIVE="${KV[THP_ENABLED_ACTIVE]:-}"
THP_DEFRAG_ACTIVE="${KV[THP_DEFRAG_ACTIVE]:-}"
[[ -z "$THP_ENABLED_ACTIVE" && -n "${KV[THP_ENABLED_RAW]:-}" ]] && THP_ENABLED_ACTIVE="$(active_from_thp_line "${KV[THP_ENABLED_RAW]}")"
[[ -z "$THP_DEFRAG_ACTIVE"   && -n "${KV[THP_DEFRAG_RAW]:-}"   ]] && THP_DEFRAG_ACTIVE="$(active_from_thp_line "${KV[THP_DEFRAG_RAW]}")"
[[ -z "$THP_ENABLED_ACTIVE" && -n "${KV[THP_ENABLED]:-}" ]] && THP_ENABLED_ACTIVE="$(active_from_thp_line "${KV[THP_ENABLED]}")"
[[ -z "$THP_DEFRAG_ACTIVE"  && -n "${KV[THP_DEFRAG]:-}"  ]] && THP_DEFRAG_ACTIVE="$(active_from_thp_line "${KV[THP_DEFRAG]}")"

# список CPU
mapfile -t CPUS < <(ls -d ${CPU_PATH}/cpu[0-9]* 2>/dev/null | sed 's#.*/cpu##' | sort -n)

echo "[restore] Restoring from: $STATE_FILE"

# 1) ASLR
if [[ -n "$ASLR" && -w "$ASLR_FILE" ]]; then
  echo "[restore] ASLR -> $ASLR"
  echo "$ASLR" > "$ASLR_FILE" || true
fi

# 2) Turbo / Boost
if [[ -e "$INTEL_TURBO" && -n "$INTEL_NO_TURBO" ]]; then
  echo "[restore] intel_pstate/no_turbo -> $INTEL_NO_TURBO"
  echo "$INTEL_NO_TURBO" > "$INTEL_TURBO" || true
elif [[ -e "$GENERIC_BOOST" && -n "$CPUFREQ_BOOST" ]]; then
  echo "[restore] cpufreq/boost -> $CPUFREQ_BOOST"
  echo "$CPUFREQ_BOOST" > "$GENERIC_BOOST" || true
fi

# 3) SMT / NMI watchdog
if [[ -n "$SMT_VAL" && -w "$SMT_CTRL" ]]; then
  echo "[restore] SMT control -> $SMT_VAL"
  echo "$SMT_VAL" > "$SMT_CTRL" || true
fi
if [[ -n "$NMI_VAL" && -w "$NMI_FILE" ]]; then
  echo "[restore] NMI watchdog -> $NMI_VAL"
  echo "$NMI_VAL" > "$NMI_FILE" || true
fi

# 4) cpupower MIN/MAX (если установлен) — берём из CPU0
if command -v cpupower >/dev/null 2>&1; then
  if [[ -n "${CPU_MIN[0]:-}" && -n "${CPU_MAX[0]:-}" ]]; then
    MIN_GHZ=$(LC_ALL=C awk -v khz="${CPU_MIN[0]}" 'BEGIN{printf "%.2fGHz", khz/1000000.0}')
    MAX_GHZ=$(LC_ALL=C awk -v khz="${CPU_MAX[0]}" 'BEGIN{printf "%.2fGHz", khz/1000000.0}')
    echo "[restore] cpupower --min $MIN_GHZ --max $MAX_GHZ"
    LC_ALL=C cpupower frequency-set --min "$MIN_GHZ" --max "$MAX_GHZ" >/dev/null || true
  fi
fi

# 5) per-CPU governor/min/max/EPP (через sysfs)
for c in "${CPUS[@]}"; do
  GOV="${CPU_PATH}/cpu${c}/cpufreq/scaling_governor"
  MIN="${CPU_PATH}/cpu${c}/cpufreq/scaling_min_freq"
  MAX="${CPU_PATH}/cpu${c}/cpufreq/scaling_max_freq"
  EPP="${CPU_PATH}/cpu${c}/cpufreq/energy_performance_preference"

  [[ -n "${CPU_GOV[$c]:-}" && -w "$GOV" ]] && { echo "[restore] cpu${c} governor -> ${CPU_GOV[$c]}"; echo "${CPU_GOV[$c]}" > "$GOV" || true; }
  [[ -n "${CPU_MIN[$c]:-}" && -w "$MIN" ]] && { echo "[restore] cpu${c} min_freq -> ${CPU_MIN[$c]}"; echo "${CPU_MIN[$c]}" > "$MIN" || true; }
  [[ -n "${CPU_MAX[$c]:-}" && -w "$MAX" ]] && { echo "[restore] cpu${c} max_freq -> ${CPU_MAX[$c]}"; echo "${CPU_MAX[$c]}" > "$MAX" || true; }
  [[ -n "${CPU_EPP[$c]:-}" && -w "$EPP" ]] && { echo "[restore] cpu${c} EPP -> ${CPU_EPP[$c]}"; echo "${CPU_EPP[$c]}" > "$EPP" || true; }
done

# 6) THP
if [[ -n "$THP_ENABLED_ACTIVE" && -w "$THP_ENABLED" ]]; then
  echo "[restore] THP enabled -> $THP_ENABLED_ACTIVE"
  echo "$THP_ENABLED_ACTIVE" > "$THP_ENABLED" || true
fi
if [[ -n "$THP_DEFRAG_ACTIVE" && -w "$THP_DEFRAG" ]]; then
  echo "[restore] THP defrag  -> $THP_DEFRAG_ACTIVE"
  echo "$THP_DEFRAG_ACTIVE" > "$THP_DEFRAG" || true
fi

# 7) C-states по CPUIDLE_SAVE_COUNT
CSTATE_COUNT="${KV[CPUIDLE_SAVE_COUNT]:-0}"
if [[ "$CSTATE_COUNT" =~ ^[0-9]+$ && "$CSTATE_COUNT" -gt 0 ]]; then
  echo "[restore] Restoring cpuidle disables ($CSTATE_COUNT items)…"
  for ((i=1; i<=CSTATE_COUNT; i++)); do
    path_key="CPUIDLE_SAVE_${i}_PATH"
    val_key="CPUIDLE_SAVE_${i}_VAL"
    path="${KV[$path_key]:-}"
    val="${KV[$val_key]:-}"
    [[ -n "$path" && -n "$val" && -w "$path" ]] || continue
    echo "  $path <- $val"
    echo "$val" > "$path" || true
  done
fi

# 8) IRQ affinities (если были сохранены)
if [[ -n "$IRQ_LIST" ]]; then
  echo "[restore] Restoring IRQ affinities…"
  for irq in $IRQ_LIST; do
    aff="${IRQ_AFF[$irq]:-}"
    AFFL="/proc/irq/${irq}/smp_affinity_list"
    [[ -n "$aff" && -w "$AFFL" ]] || continue
    echo "  IRQ $irq <- $aff"
    echo "$aff" > "$AFFL" || true
  done
fi

echo
echo "[restore] Summary:"
command -v cpupower >/dev/null 2>&1 && cpupower frequency-info | sed -n '1,25p' || true
[[ -e "$INTEL_TURBO"   ]] && echo "[restore] intel_pstate/no_turbo = $(cat "$INTEL_TURBO" 2>/dev/null || echo '?')" || true
[[ -e "$GENERIC_BOOST" ]] && echo "[restore] cpufreq/boost        = $(cat "$GENERIC_BOOST" 2>/dev/null || echo '?')" || true
[[ -e "$ASLR_FILE"     ]] && echo "[restore] ASLR                   = $(cat "$ASLR_FILE" 2>/dev/null || echo '?')" || true
[[ -e "$THP_ENABLED"   ]] && echo "[restore] THP enabled            = $(cat "$THP_ENABLED" 2>/dev/null || echo '?')" || true
[[ -e "$THP_DEFRAG"    ]] && echo "[restore] THP defrag             = $(cat "$THP_DEFRAG" 2>/dev/null || echo '?')" || true
[[ -e "$SMT_CTRL"      ]] && echo "[restore] SMT control            = $(cat "$SMT_CTRL" 2>/dev/null || echo '?')" || true
[[ -e "$NMI_FILE"      ]] && echo "[restore] NMI watchdog           = $(cat "$NMI_FILE" 2>/dev/null || echo '?')" || true
