#!/bin/bash


## Need to silence

# Source a file that provides SEMTools
#source $1

# SESSRECORDS="PHD_024/0093/88775 PHD_024/0093/60307 PHD_024/0132/74443"
BASE_OUTDIR=/Shared/johnsonhj/HDNI/20131216_RobotRater
OUTDIR=${BASE_OUTDIR}/quick_scripts

mkdir -p ${OUTDIR}

for SESSRECORD in $(cat scanids); do
  session=$(echo ${SESSRECORD} |awk -F/ '{print $3}')
  thisScript=${OUTDIR}/Do_${session}.sh
  cat > ${thisScript} << EOF
module load gcc/4.8.2

export PATH=/Shared/sinapse/sharedopt/20131115/RHEL6/NEP-build/bin:${PATH}
source /Shared/sinapse/sharedopt/20131009/RHEL6/anaconda-bin/bin/activate nipype_env
export PYTHONPATH=/Shared/sinapse/sharedopt/20131115/RHEL6/NEP-build/lib:/Shared/sinapse/sharedopt/20131009/src/NAMICExternalProjects/SuperBuild/ExternalSources/BRAINSTools/AutoWorkup:/Shared/sinapse/sharedopt/20131009/src/NAMICExternalProjects/SuperBuild/ExternalSources/NIPYPE/

DATADIR=/Shared/paulsen/Experiments
BENCHMARK=\${DATADIR}/20130202_PREDICTHD_Results
TEST=\${DATADIR}/20131124_PREDICTHD_Results
which python
python ${BASE_OUTDIR}/robotRater.py -b \${BENCHMARK} -t \${TEST} -s ${SESSRECORD} -o ${OUTDIR} -r ${session}.csv
EOF
  chmod ug+rwx ${thisScript}
  if [ ! -f ${OUTDIR}/${session}.csv ]; then
      qsub -q all.q -pe smp 1-4 ${thisScript} $1>>seriesRun.log $2>>seriesRun.err
  else
      echo "Done: $session"
  fi
done
