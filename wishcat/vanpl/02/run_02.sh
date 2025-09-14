for i in {1..10}; do
  python3 user_feature_build.py --input users_logs.json --days 30 \
    --output feature_u_demo1_$i.json --log run_$i.csv
done
