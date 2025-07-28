#!/bin/bash

# 입력 디렉토리와 이동할 디렉토리 설정
SOURCE_DIR=/home/hyunsoocha/GitHub/perse_before_release/data/datasets/guy/synthetic_dataset
TARGET_DIR=/home/hyunsoocha/GitHub/perse_before_release/data/datasets/guy/synthetic_dataset_others
LAST_PATTERN=guy_hyunsoo_img_6948
LIMIT=20

# TARGET_DIR이 존재하지 않으면 생성
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# 디렉토리 이름에서 attribute를 추출하는 패턴 설정
PATTERN="^[0-9]+_([a-zA-Z]+)_.+_${LAST_PATTERN}$"

# SOURCE_DIR 내의 모든 디렉토리 처리
for attr in $(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -Eo "^[0-9]+_[a-zA-Z]+_" | cut -d'_' -f2 | sort | uniq)
do
    echo "Processing attribute: $attr"

    # 해당 attribute의 디렉토리를 정렬하여 리스트 생성
    DIRS=$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
        | grep -E "^[0-9]+_${attr}_.+_${LAST_PATTERN}$" \
        | sort -t'_' -k1,1n)

    COUNT=0

    for dir in $DIRS
    do
        FULL_PATH="$SOURCE_DIR/$dir"

        if [ $COUNT -lt $LIMIT ]; then
            echo "Keeping: $dir"
        else
            echo "Moving: $dir to $TARGET_DIR"
            mv "$FULL_PATH" "$TARGET_DIR/"
        fi

        COUNT=$((COUNT + 1))
    done
done

echo "Done."