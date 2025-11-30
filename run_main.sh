#!/bin/bash

# Script to run main.py with proper environment setup
# Usage: ./run_main.sh

cd "$(dirname "$0")"

echo "Running BLUEmed main.py..."
echo ""

# Pass note content via stdin to avoid interactive input issues
python main.py << 'EOF'
54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
along with pitting edema and dilated veins. Diagnosed as a venous ulcer.

EOF
