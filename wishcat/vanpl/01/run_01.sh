for i in {1..10}; do 
  python3 marker_process.py --input sample.json --output result_$i.json --log run_$i.log 
done
