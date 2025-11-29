#!/bin/bash
API=http://localhost:8000
FILENAME=ted2.mp4
SEGMENT_COUNT=3              # сколько лучших сегментов брать автоматически
VERTICAL_METHOD=letterbox

die() { echo "$1" >&2; exit 1; }

START=$(date +%s)

# --- Анализ ---
TASK_ANALYZE=$(curl -s -X POST "$API/api/video/analyze-local?filename=$FILENAME" | jq -r '.task_id')
[ -n "$TASK_ANALYZE" ] && [ "$TASK_ANALYZE" != "null" ] || die "Не получил task_id анализа"

echo "Анализ task: $TASK_ANALYZE"
while true; do
  STATUS_JSON=$(curl -s "$API/api/video/task/$TASK_ANALYZE")
  STATUS=$(echo "$STATUS_JSON" | jq -r '.status')
  MSG=$(echo "$STATUS_JSON" | jq -r '.message')
  printf "\r[analysis %s] %s" "$STATUS" "$MSG"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && { echo; die "Анализ упал"; }
  sleep 5
done
echo

VIDEO_ID=$(echo "$STATUS_JSON" | jq -r '.result.video_id')
SEGMENTS_JSON=$(echo "$STATUS_JSON" | jq '.result.segments')

# авто-выбор топовых сегментов по score
SEGMENT_IDS=$(echo "$SEGMENTS_JSON" \
  | jq -r "sort_by(-.highlight_score)[:$SEGMENT_COUNT][] .id")
echo "Выбраны сегменты: $SEGMENT_IDS"

# --- Рендер ---
PAYLOAD=$(jq -n \
  --arg vid "$VIDEO_ID" \
  --arg method "$VERTICAL_METHOD" \
  --argjson segs "$(printf '%s\n' "$SEGMENT_IDS" | jq -R 'select(length>0)' | jq -s '.')" \
  '{video_id:$vid, vertical_method:$method, segment_ids:$segs}')

TASK_PROCESS=$(curl -s -X POST "$API/api/video/process" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | jq -r '.task_id')
[ -n "$TASK_PROCESS" ] && [ "$TASK_PROCESS" != "null" ] || die "Не получил task_id обработки"

echo "Обработка task: $TASK_PROCESS"
while true; do
  STATUS_JSON=$(curl -s "$API/api/video/task/$TASK_PROCESS")
  STATUS=$(echo "$STATUS_JSON" | jq -r '.status')
  MSG=$(echo "$STATUS_JSON" | jq -r '.message')
  printf "\r[process %s] %s" "$STATUS" "$MSG"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && { echo; die "Обработка упала"; }
  sleep 5
done
echo

END=$(date +%s)
ELAPSED=$((END-START))
printf "Полный цикл: %s сек (%s)\n" "$ELAPSED" "$(date -u -d @${ELAPSED} +%H:%M:%S)"
