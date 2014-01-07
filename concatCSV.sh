#!/bin/bash
OUTPUT=concatCSV.csv
if [ -f ${OUTPUT}.err ]; then
    rm ${OUTPUT}.err
fi

HEADER="session,structure,volume,hausdorff,hausdorffAvg,dice,border,benchmark,test"
echo ${HEADER} > ${OUTPUT}

for file in $(cat csvfiles); do
    SESSION=$(echo ${file} | awk -F/ '{print $4}')
    echo $SESSION
    if [ ! -f ${file} ]; then
        echo ${file} >> ${OUTPUT}.err
    else
        for line in $(cat ${file}); do
            if [ ${line:0:9} != 'structure' ]; then 
                echo ${SESSION},${line} >> ${OUTPUT}
            fi
        done
    fi
done
