OUTPUT_FOLDER=$1
head -1 $(ls $OUTPUT_FOLDER/*_pq*.csv | head -1) > full.csv
ls output*/*_pq*.csv | awk -F/ '{print $1}' | sort -u | while read d
do
cat ${d}/*pq*.csv | grep -v dataset_name >> full.csv
done
