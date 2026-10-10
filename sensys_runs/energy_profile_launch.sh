#!/usr/bin/env bash
# Static shards only; all audit artifacts are outside the repository.
set -euo pipefail
die() { printf 'STOP: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 3 || ( $# -eq 4 && $4 == --dry-run ) ]] || die 'usage: energy_profile_launch.sh MANIFEST K SESSION [--dry-run]'
manifest=$1
count=$2
session=$3
dry=${4:-}
topic=${NTFY_TOPIC:-}
[[ -n ${topic//[[:space:]]/} ]] || die 'NTFY_TOPIC is required'
[[ $count =~ ^[1-9][0-9]*$ ]] || die 'K must be a positive integer'
[[ $session =~ ^[a-zA-Z0-9_-]+$ ]] || die 'invalid tmux session name'
command -v tmux >/dev/null || die 'tmux is required'
if tmux has-session -t "=$session" 2>/dev/null; then die 'tmux session already exists'; fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python="$HOME/miniconda3/envs/LIO_tecs/bin/python"
runner="$repo/sensys_runs/energy_profile_runner.py"
[[ -x $python ]] || die "missing interpreter: $python"
# Resolve metadata without regenerating any training argv.
metadata=$("$python" -B -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); m=json.loads(p.read_text()); print(p); print(m["threads"]); print(pathlib.Path.home()/"iccps_runs"/(m["campaign"]+"_console"))' "$manifest")
mapfile -t fields <<< "$metadata"
manifest=${fields[0]}
threads=${fields[1]}
console=${fields[2]}
[[ $threads == 1 ]] || die 'manifest threads must be 1'
(( count * threads <= $(nproc) )) || die 'K x threads exceeds nproc'
[[ ! -e $console && ! -L $console ]] || die "console directory must be absent: $console"
# This read-only gate also rejects dirty git, stale targets and unignored results.
"$python" -B "$runner" "$manifest" --shard 0 --of "$count" --dry-run >/dev/null
token=$("$python" -B -c 'import uuid; print(uuid.uuid4().hex)')
if [[ -z $dry ]]; then free -h; mkdir -p -- "$(dirname -- "$console")"; mkdir -- "$console"; fi
for ((i=0; i<count; i++)); do
    # A pre-existing tmux server may have stale notification environment values.
    printf -v command '%q ' env "NTFY_TOPIC=$NTFY_TOPIC" "NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}" "$python" -B "$runner" "$manifest" --shard "$i" --of "$count" --console-dir "$console" --start-together "$token"
    # tmux executes a Bash-quoted command; the runner executes argv without a shell.
    printf -v command 'exec bash -c %q' "$command"
    if (( i == 0 )); then
        launch=(tmux new-session -d -s "$session" -n "shard-$i" "$command")
    else
        launch=(tmux new-window -t "$session:" -n "shard-$i" "$command")
    fi
    if [[ -n $dry ]]; then printf '%q ' "${launch[@]}"; printf '\n'; else "${launch[@]}"; fi
done
if [[ -z $dry ]]; then
    printf 'tmux attach -t %q\n' "$session"
    printf 'tail -F %q/shard-*.log\n' "$console"
fi
