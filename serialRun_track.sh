#!/bin/bash

# Source a file that provides SEMTools
#source $1

BASE_OUTDIR=/Shared/johnsonhj/HDNI/20131216_RobotRater
OUTDIR=${BASE_OUTDIR}/quick_scripts

mkdir -p ${OUTDIR}

for SESSRECORD in $(cat track_scanids.list); do  #scanids); do
  session=$(echo ${SESSRECORD} | awk -F/ '{print $3}')
  thisScript=${OUTDIR}/Do_${session}.sh
  cat > ${thisScript} << EOF
#!/bin/bash
module load gcc/4.8.2

export PATH=/Shared/sinapse/sharedopt/20131115/RHEL6/NEP-build/bin:${PATH}
source /Shared/sinapse/sharedopt/20131009/RHEL6/anaconda-bin/bin/activate nipype_env
export PYTHONPATH=/Shared/sinapse/sharedopt/20131115/RHEL6/NEP-build/lib:/Shared/sinapse/sharedopt/20131009/src/NAMICExternalProjects/SuperBuild/ExternalSources/BRAINSTools/AutoWorkup:/Shared/sinapse/sharedopt/20131009/src/NAMICExternalProjects/SuperBuild/ExternalSources/NIPYPE/:${PYTHONPATH}

DATADIR=/Shared/johnsonhj/TrackOn/Experiments
BENCHMARK=\${DATADIR}/20130109_TrackOn_Results
TEST=\${DATADIR}/20131119_TrackOn_Results

python ${BASE_OUTDIR}/roboRater.py -b \${BENCHMARK} -t \${TEST} -s ${SESSRECORD} -o ${OUTDIR} -r ${session}.csv
EOF

  chmod ug+rwx ${thisScript}
  if [[ ! -f ${OUTDIR}/${SESSRECORD}/${session}.csv ]]; then
      qsub -q all.q -pe smp 1-4 ${thisScript}
  else
      echo "Done: $session"
  fi
done
