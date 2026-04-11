#!/bin/bash
# prepare-light-dataset.sh
# Wrapper script for preparing lightweight dataset on spectrum

set -euo pipefail

# Default values
SOURCE_PATH="${1:- }"
DEST_PATH="${2:- }"
DRY_RUN="${3:-false}"
FILES_PER_TYPE=300
SEED=42

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 SOURCE_PATH [DEST_PATH] [--dry-run]"
    echo ""
    echo "Arguments:"
    echo "  SOURCE_PATH    Path to source extracted folder"
    echo "  DEST_PATH      Path for extracted-light (default: same parent)"
    echo "  --dry-run      Preview without copying"
    echo ""
    echo "Examples:"
    echo "  # Dry run (preview)"
    echo "  $0 /home/n-salikhova/datasets/extracted/Celeb-DF-v3 --dry-run"
    echo ""
    echo "  # Full run"
    echo "  $0 /home/n-salikhova/datasets/extracted/Celeb-DF-v3"
    echo ""
}

if [[ -z "$SOURCE_PATH" ]]; then
    echo -e "${RED}Error: SOURCE_PATH required${NC}"
    print_usage
    exit 1
fi

# If DEST_PATH not provided, use parent/extracted-light
if [[ -z "$DEST_PATH" ]]; then
    PARENT=$(dirname "$SOURCE_PATH")
    DEST_PATH="${PARENT}/extracted-light"
fi

# Check for --dry-run flag
if [[ "$DEST_PATH" == "--dry-run" ]]; then
    DRY_RUN="true"
    PARENT=$(dirname "$SOURCE_PATH")
    DEST_PATH="${PARENT}/extracted-light"
elif [[ "$DRY_RUN" == "--dry-run" ]]; then
    DRY_RUN="true"
fi

# Validate source
if [[ ! -d "$SOURCE_PATH" ]]; then
    echo -e "${RED}Error: Source path does not exist or is not a directory${NC}"
    echo "  $SOURCE_PATH"
    exit 1
fi

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Prepare Light Dataset${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  SOURCE:           $SOURCE_PATH"
echo "  DESTINATION:      $DEST_PATH"
echo "  FILES PER TYPE:   $FILES_PER_TYPE"
echo "  SEED:             $SEED"
echo "  DRY RUN:          $DRY_RUN"
echo ""

# Build command
CMD="python3 scripts/prepare_light_dataset.py"
CMD="$CMD --source '$SOURCE_PATH'"
CMD="$CMD --dest '$DEST_PATH'"
CMD="$CMD --files-per-type $FILES_PER_TYPE"
CMD="$CMD --seed $SEED"

if [[ "$DRY_RUN" == "true" ]]; then
    CMD="$CMD --dry-run"
    echo -e "${YELLOW}Running in DRY-RUN mode (no files will be copied)${NC}"
    echo ""
fi

echo -e "${BLUE}Starting build...${NC}"
echo ""

# Run the Python script
eval "$CMD"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ Success!${NC}"
    if [[ "$DRY_RUN" != "true" ]]; then
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo "  1. Verify extracted-light folder:"
        echo "     du -sh $DEST_PATH"
        echo "  2. Create tar.gz archive (optional):"
        echo "     tar -czf extracted-light.tar.gz -C $(dirname "$DEST_PATH") extracted-light/"
    fi
else
    echo -e "${RED}✗ Failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
