#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="$HOME/.bench_env"
STATE_FILE="$STATE_DIR/state.env"
mkdir -p "$STATE_DIR"
: > "$STATE_FILE"

echo "[prepare] Saving current settings to: $STATE_FILE"

CPU_PATH="/sys/devices/system/cpu"
mapfile -t CPUS < <(ls -d ${CPU_PATH}/cpu[0-9]* 2>/dev/null | sed 's#^.*/cpu##' | sort -n)

# Save per-CPU governor/min/max/EPP
for c in "${CPUS[@]}"; do
  GOV="${CPU_PATH}/cpu${c}/cpufreq/scaling_governor"
  MIN="${CPU_PATH}/cpu${c}/cpufreq/scaling_min_freq"
  MAX="${CPU_PATH}/cpu${c}/cpufreq/scaling_max_freq"
  EPP="${CPU_PATH}/cpu${c}/cpufreq/energy_performance_preference"
  [[ -r "$GOV" ]] && echo "CPU${c}_GOV=$(cat "$GOV")" >> "$STATE_FILE"
  [[ -r "$MIN" ]] && echo "CPU${c}_MIN=$(cat "$MIN")" >> "$STATE_FILE"
  [[ -r "$MAX" ]] && echo "CPU${c}_MAX=$(cat "$MAX")" >> "$STATE_FILE"
  [[ -r "$EPP" ]] && echo "CPU${c}_EPP=$(cat "$EPP")" >> "$STATE_FILE"
done

# Save Turbo/Boost
INTEL_TURBO="/sys/devices/system/cpu/intel_pstate/no_turbo"
GENERIC_BOOST="/sys/devices/system/cpu/cpufreq/boost"
if [[ -e "$INTEL_TURBO" ]]; then
  echo "INTEL_NO_TURBO=$(cat "$INTEL_TURBO")" >> "$STATE_FILE"
elif [[ -e "$GENERIC_BOOST" ]]; then
  echo "CPUFREQ_BOOST=$(cat "$GENERIC_BOOST")" >> "$STATE_FILE"
fi

# Save ASLR
ASLR_FILE="/proc/sys/kernel/randomize_va_space"
echo "ASLR=$(cat "$ASLR_FILE")" >> "$STATE_FILE"

# --- Save THP (Transparent Huge Pages) mode & defrag ---
THP_ENABLED="/sys/kernel/mm/transparent_hugepage/enabled"
THP_DEFRAG="/sys/kernel/mm/transparent_hugepage/defrag"

if [[ -r "$THP_ENABLED" ]]; then
  THP_ENABLED_RAW=$(cat "$THP_ENABLED")
  THP_ENABLED_ACTIVE=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<<"$THP_ENABLED_RAW")
  [[ -z "$THP_ENABLED_ACTIVE" ]] && THP_ENABLED_ACTIVE="$THP_ENABLED_RAW"
  echo "THP_ENABLED_RAW=$THP_ENABLED_RAW"           >> "$STATE_FILE"
  echo "THP_ENABLED_ACTIVE=$THP_ENABLED_ACTIVE"     >> "$STATE_FILE"
fi
if [[ -r "$THP_DEFRAG" ]]; then
  THP_DEFRAG_RAW=$(cat "$THP_DEFRAG")
  THP_DEFRAG_ACTIVE=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<<"$THP_DEFRAG_RAW")
  [[ -z "$THP_DEFRAG_ACTIVE" ]] && THP_DEFRAG_ACTIVE="$THP_DEFRAG_RAW"
  echo "THP_DEFRAG_RAW=$THP_DEFRAG_RAW"             >> "$STATE_FILE"
  echo "THP_DEFRAG_ACTIVE=$THP_DEFRAG_ACTIVE"       >> "$STATE_FILE"
fi

# --- Save per-CPU cpuidle state disable flags (C-states) ---
for c in "${CPUS[@]}"; do
  for sdir in "${CPU_PATH}/cpu${c}/cpuidle"/state*; do
    [[ -r "$sdir/disable" ]] || continue
    sid=$(basename "$sdir" | sed 's/state//')
    val=$(cat "$sdir/disable")
    echo "CPU${c}_CSTATE${sid}_DIS=${val}" >> "$STATE_FILE"
  done
done

# --- Save IRQ affinities (list + each smp_affinity_list) ---
mapfile -t IRQS < <(ls -d /proc/irq/[0-9]* 2>/dev/null | sed 's#.*/##' | sort -n)
echo -n "IRQ_LIST=" >> "$STATE_FILE"; printf "%s " "${IRQS[@]}" >> "$STATE_FILE"; echo >> "$STATE_FILE"
for irq in "${IRQS[@]}"; do
  AFFL="/proc/irq/${irq}/smp_affinity_list"
  [[ -r "$AFFL" ]] && echo "IRQ${irq}_AFFINITY_LIST=$(cat "$AFFL")" >> "$STATE_FILE"
done


echo "[prepare] Saved snapshot:"
cat "$STATE_FILE"
echo

echo "[prepare] Applying deterministic settings..."

# 1) Disable Turbo/Boost
if [[ -e "$INTEL_TURBO" ]]; then
  echo "[prepare] Disabling Intel Turbo (no_turbo=1)"
  echo 1 | sudo tee "$INTEL_TURBO" >/dev/null
elif [[ -e "$GENERIC_BOOST" ]]; then
  echo "[prepare] Disabling cpufreq boost (0)"
  echo 0 | sudo tee "$GENERIC_BOOST" >/dev/null
else
  echo "[prepare][info] Turbo/boost control not found; skipping"
fi

# 2) Governor = performance
if command -v cpupower >/dev/null 2>&1; then
  echo "[prepare] Setting governor=performance (all CPUs)"
  sudo cpupower frequency-set --governor performance >/dev/null || true
else
  echo "[prepare][warn] cpupower not found, using sysfs for governor"
  for c in "${CPUS[@]}"; do
    GOV="${CPU_PATH}/cpu${c}/cpufreq/scaling_governor"
    [[ -w "$GOV" ]] && echo performance | sudo tee "$GOV" >/dev/null
  done
fi

# 3) Choose target frequency = base (not turbo)
BASE_FILE="${CPU_PATH}/cpu0/cpufreq/base_frequency"
MAX_FILE0="${CPU_PATH}/cpu0/cpufreq/scaling_max_freq"
CPUINFO_MAX="${CPU_PATH}/cpu0/cpufreq/cpuinfo_max_freq"

TARGET_KHZ=""
if [[ -r "$BASE_FILE" ]]; then
  TARGET_KHZ=$(cat "$BASE_FILE")
  echo "[prepare] base_frequency detected: ${TARGET_KHZ} kHz"
elif [[ -r "$MAX_FILE0" ]]; then
  TARGET_KHZ=$(cat "$MAX_FILE0")
  echo "[prepare] using current scaling_max_freq: ${TARGET_KHZ} kHz"
elif [[ -r "$CPUINFO_MAX" ]]; then
  TARGET_KHZ=$(cat "$CPUINFO_MAX")
  echo "[prepare] cpuinfo_max_freq used (may be turbo): ${TARGET_KHZ} kHz"
fi

# 4) Lock min=max to TARGET_KHZ
if [[ -n "$TARGET_KHZ" ]]; then
  # via cpupower (force C locale for decimal point)
  if command -v cpupower >/dev/null 2>&1; then
    GHZ=$(LC_ALL=C awk -v khz="$TARGET_KHZ" 'BEGIN{printf "%.2fGHz", khz/1000000.0}')
    echo "[prepare] Locking via cpupower --min $GHZ --max $GHZ"
    LC_ALL=C sudo cpupower frequency-set --min "$GHZ" --max "$GHZ" >/dev/null || true
  fi
  # via sysfs per CPU
    for c in "${CPUS[@]}"; do
    MIN="${CPU_PATH}/cpu${c}/cpufreq/scaling_min_freq"
    MAX="${CPU_PATH}/cpu${c}/cpufreq/scaling_max_freq"
    [[ -e "$MIN" ]] && echo "$TARGET_KHZ" | sudo tee "$MIN" >/dev/null
    [[ -e "$MAX" ]] && echo "$TARGET_KHZ" | sudo tee "$MAX" >/dev/null
    done
else
  echo "[prepare][warn] No frequency files found; skipping hard lock"
fi

# 5) EPP → performance (if available)
for c in "${CPUS[@]}"; do
  EPP="${CPU_PATH}/cpu${c}/cpufreq/energy_performance_preference"
  [[ -e "$EPP" ]] && echo performance | sudo tee "$EPP" >/dev/null
done

# 6) Disable ASLR (per assignment)
echo "[prepare] Disabling ASLR (randomize_va_space=0)"
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space >/dev/null


# 7) Limit CPU idle states (disable deep C-states)
echo "[prepare] Disabling CPU idle states (cpuidle)"
for c in "${CPUS[@]}"; do
  for sdir in "${CPU_PATH}/cpu${c}/cpuidle"/state*; do
    [[ -w "$sdir/disable" ]] || continue
    echo 1 | sudo tee "$sdir/disable" >/dev/null || true
  done
done

# 8) Disable Transparent Huge Pages (THP)
THP_ENABLED="/sys/kernel/mm/transparent_hugepage/enabled"
THP_DEFRAG="/sys/kernel/mm/transparent_hugepage/defrag"
if [[ -w "$THP_ENABLED" ]]; then
  echo "[prepare] Disabling THP (enabled=never)"
  echo never | sudo tee "$THP_ENABLED" >/dev/null || true
fi
if [[ -w "$THP_DEFRAG" ]]; then
  echo "[prepare] Disabling THP defrag (defrag=never)"
  echo never | sudo tee "$THP_DEFRAG" >/dev/null || true
fi

# 9) Migrate IRQs away from the benchmark CPU
# Используй переменную окружения BENCH_CPU (по умолчанию 2)
BENCH_CPU="${BENCH_CPU:-2}"
echo "[prepare] Moving IRQs off CPU ${BENCH_CPU}"

# Список всех IRQ
mapfile -t IRQS < <(ls -d /proc/irq/[0-9]* 2>/dev/null | sed 's#.*/##' | sort -n)

# Построить affinity_list без целевого CPU: "0,1,3,4,..."
CPU_LIST_NO_TARGET=""
for cpu in "${CPUS[@]}"; do
  [[ "$cpu" == "$BENCH_CPU" ]] && continue
  CPU_LIST_NO_TARGET="${CPU_LIST_NO_TARGET:+$CPU_LIST_NO_TARGET,}${cpu}"
done

for irq in "${IRQS[@]}"; do
  AFFL="/proc/irq/${irq}/smp_affinity_list"
  [[ -w "$AFFL" ]] || continue
  echo "$CPU_LIST_NO_TARGET" | sudo tee "$AFFL" >/dev/null || true
done


echo
echo "[prepare] Summary:"
if command -v cpupower >/dev/null 2>&1; then
  cpupower frequency-info | sed -n '1,25p'
fi
if [[ -e "$INTEL_TURBO" ]]; then
  echo "[prepare] intel_pstate/no_turbo = $(cat "$INTEL_TURBO")"
elif [[ -e "$GENERIC_BOOST" ]]; then
  echo "[prepare] cpufreq/boost        = $(cat "$GENERIC_BOOST")"
fi
echo "[prepare] ASLR                   = $(cat /proc/sys/kernel/randomize_va_space)"

echo
echo "[prepare] Tip: pin your workload, e.g."
echo "  taskset -c 2 ./build/bench_insert --benchmark_filter='BM_Insert_Average/32768' --benchmark_min_time=2s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true"
