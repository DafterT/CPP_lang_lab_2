#!/usr/bin/env bash
# restore_bench_env.sh
# Восстановление "стандартных" настроек из вашего лога-снапшота.
# Требуются права root.

set -euo pipefail

echo "[restore] Starting…"

CPU_PATH="/sys/devices/system/cpu"
CPUS=($(ls -d ${CPU_PATH}/cpu[0-9]* 2>/dev/null | sed 's#^.*/cpu##' | sort -n))

# ---- Параметры из снапшота ----
GOV="powersave"
MIN_KHZ="400000"
MAX_KHZ="4200000"
EPP="balance_performance"
ASLR_VAL="2"                 # включённый ASLR
NO_TURBO="0"                 # 0 = Turbo включён
SMT_VAL="on"
NMI_WATCHDOG_VAL="1"
THP_MODE="madvise"           # /sys/kernel/mm/transparent_hugepage/enabled
THP_DEFRAG_MODE="madvise"    # /sys/kernel/mm/transparent_hugepage/defrag

# ---- Утилита безопасной записи ----
w() {
  local val="$1" path="$2"
  if [[ -e "$path" && -w "$path" ]]; then
    echo "$val" > "$path" || true
  fi
}

# ---- CPU freq / governor / EPP ----
echo "[restore] Setting governor=${GOV}, min=${MIN_KHZ}, max=${MAX_KHZ}, EPP=${EPP}"
for c in "${CPUS[@]}"; do
  GOV_PATH="${CPU_PATH}/cpu${c}/cpufreq/scaling_governor"
  MIN_PATH="${CPU_PATH}/cpu${c}/cpufreq/scaling_min_freq"
  MAX_PATH="${CPU_PATH}/cpu${c}/cpufreq/scaling_max_freq"
  EPP_PATH="${CPU_PATH}/cpu${c}/cpufreq/energy_performance_preference"

  # Порядок: сначала governor, затем границы частоты
  w "$GOV" "$GOV_PATH"
  w "$MIN_KHZ" "$MIN_PATH"
  w "$MAX_KHZ" "$MAX_PATH"
  # EPP не у всех доступен, пропускаем если файла нет
  w "$EPP" "$EPP_PATH"
done

# ---- Intel Turbo ----
INTEL_NO_TURBO_PATH="${CPU_PATH}/intel_pstate/no_turbo"
if [[ -e "$INTEL_NO_TURBO_PATH" ]]; then
  echo "[restore] Enabling Intel Turbo (no_turbo=${NO_TURBO})"
  w "$NO_TURBO" "$INTEL_NO_TURBO_PATH"
fi

# ---- ASLR ----
if [[ -e /proc/sys/kernel/randomize_va_space ]]; then
  echo "[restore] ASLR=${ASLR_VAL}"
  w "$ASLR_VAL" /proc/sys/kernel/randomize_va_space
fi

# ---- Включить глубокие C-states (disable=0 для всех state*/disable) ----
echo "[restore] Enabling deep C-states (cpuidle state*/disable=0)"
for c in "${CPUS[@]}"; do
  for d in "${CPU_PATH}/cpu${c}/cpuidle/state"*/disable; do
    [[ -e "$d" ]] && w "0" "$d"
  done
done

# ---- Transparent Huge Pages ----
if [[ -d /sys/kernel/mm/transparent_hugepage ]]; then
  echo "[restore] THP=${THP_MODE}, THP defrag=${THP_DEFRAG_MODE}"
  w "$THP_MODE" /sys/kernel/mm/transparent_hugepage/enabled
  w "$THP_DEFRAG_MODE" /sys/kernel/mm/transparent_hugepage/defrag
fi

# ---- SMT on ----
SMT_CTRL="${CPU_PATH}/smt/control"
if [[ -e "$SMT_CTRL" ]]; then
  echo "[restore] SMT=${SMT_VAL}"
  w "$SMT_VAL" "$SMT_CTRL"
fi

# ---- NMI watchdog ----
if [[ -e /proc/sys/kernel/nmi_watchdog ]]; then
  echo "[restore] nmi_watchdog=${NMI_WATCHDOG_VAL}"
  w "$NMI_WATCHDOG_VAL" /proc/sys/kernel/nmi_watchdog
fi

# ---- Сброс IRQ аффинити: разрешить все CPU ----
# Вычисляем маску вида 0xff для числа логических CPU.
ONLINE_CPUS=$(<"${CPU_PATH}/online" tr -d '\n' || echo "")
# Фоллбэк, если файл отсутствует:
NUM_CPUS=${#CPUS[@]}
if [[ -n "$ONLINE_CPUS" && "$ONLINE_CPUS" =~ ^[0-9,-]+$ ]]; then
  # Оценим количество по nproc, чтобы не парсить сложные диапазоны
  NUM_CPUS="$(nproc --all 2>/dev/null || echo ${#CPUS[@]})"
fi
# Ограничим до 63, чтобы не лезть в многоразрядные маски (достаточно для вашего случая 0-7)
if (( NUM_CPUS > 63 )); then NUM_CPUS=63; fi
MASK_DEC=$(( (1<<NUM_CPUS) - 1 ))
MASK_HEX=$(printf "%x\n" "$MASK_DEC")

echo "[restore] Resetting IRQ affinities to mask 0x${MASK_HEX} (all CPUs)"
shopt -s nullglob
for f in /proc/irq/*/smp_affinity; do
  if [[ -w "$f" ]]; then
    echo "$MASK_HEX" > "$f" 2>/dev/null || true
  fi
done
for f in /proc/irq/*/smp_affinity_list; do
  if [[ -w "$f" ]]; then
    # Альтернативная запись через список ядер
    seq_list=$(seq -s, 0 $((NUM_CPUS-1)))
    echo "$seq_list" > "$f" 2>/dev/null || true
  fi
done

echo "[restore] Done."

# ---- Краткая сводка ----
echo "----------------------------------------"
echo "Governor:      ${GOV}"
echo "Min/Max (kHz): ${MIN_KHZ}/${MAX_KHZ}"
echo "EPP:           ${EPP}"
echo "Turbo:         $( [[ -e $INTEL_NO_TURBO_PATH ]] && cat $INTEL_NO_TURBO_PATH | sed 's/0/ON/;s/1/OFF/' || echo 'N/A')"
echo "ASLR:          $(cat /proc/sys/kernel/randomize_va_space 2>/dev/null || echo 'N/A')"
echo "SMT:           $( [[ -e $SMT_CTRL ]] && cat $SMT_CTRL || echo 'N/A')"
echo "NMI watchdog:  $(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo 'N/A')"
if [[ -d /sys/kernel/mm/transparent_hugepage ]]; then
  echo "THP enabled:   $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  echo "THP defrag:    $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
fi
echo "----------------------------------------"
