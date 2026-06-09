ruff check  --select I $1
ruff check  --select I $1 --fix
echo "-------------------------------"
ruff check $1
ruff check $1 --fix