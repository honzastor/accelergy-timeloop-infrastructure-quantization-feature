#!/bin/bash
# RUN ALL TIMELOOP MAPPER
NETWORK="MobileNet"
THREADS="all_threads"   # choices are: "1_thread", "all_threads" 

# Simba-like (only 1 chip from original simba chiplet structure, architecture similar to NVDLA), configured on 45nm technology using 32-bit float data and word widths, the architecture contains 16 PEs

mkdir -p ../outputs
for mapper in ../mapper/$THREADS/*/*.yaml; do
    for problem in ../prob/$NETWORK/*/*.yaml; do
        start_time=`date +%s`
        timeloop-mapper ../arch/simba_like.yaml ../arch/components/*.yaml ../constraints/*.yaml $mapper $problem -o ../outputs
        end_time=`date +%s`
        runtime=$((end_time - start_time))

        # Rename and store the generated csv/txt stats files and the best mapping    
        layer_dir=$(echo "$problem" | cut -d'/' -f4)
        layer_basefname="$(basename -- $problem)"
        layer_name=$(echo "$layer_basefname" | cut -d "." -f1)
        mapper_type=$(echo $mapper | cut -d '/' -f 4)
        tmp=$(echo "$(basename -- $mapper)" | cut -d '_' -f 1)
        mapper_basename="${tmp%.*}"

        oaves_dir=../outputs/oaves/$THREADS/$mapper_type/$mapper_basename/$NETWORK/$layer_dir
        csv_dir=../outputs/csv/$THREADS/$mapper_type/$mapper_basename/$NETWORK/$layer_dir
        xml_dir=../outputs/xml/$THREADS/$mapper_type/$mapper_basename/$NETWORK/$layer_dir
        txt_dir=../outputs/txt/$THREADS/$mapper_type/$mapper_basename/$NETWORK/$layer_dir
        map_dir=../outputs/map/$THREADS/$mapper_type/$mapper_basename/$NETWORK/$layer_dir
        
        mkdir -p $oaves_dir
        mkdir -p $csv_dir
        mkdir -p $xml_dir
        mkdir -p $txt_dir
        mkdir -p $map_dir
        out_fname=${layer_basefname%.*}

        # Add the execution time as a new field to the existing CSV file
        awk -v runtime=$runtime -F, 'NR==1 { print $0 ",Execution runtime [s]"; next } { print $0 "," runtime }' ../outputs/timeloop-mapper.stats.csv > ../outputs/timeloop-mapper.stats_w_time.csv
        
        mv ../outputs/timeloop-mapper.oaves.csv $oaves_dir/$out_fname.oaves.csv
        mv ../outputs/timeloop-mapper.stats_w_time.csv $csv_dir/$out_fname.stats.csv
        mv ../outputs/timeloop-mapper.map+stats.xml $xml_dir/$out_fname.map+stats.xml
        mv ../outputs/timeloop-mapper.stats.txt $txt_dir/$out_fname.stats.txt
        mv ../outputs/timeloop-mapper.map.yaml $map_dir/$out_fname.map.yaml
    done
done