#!/bin/bash

# PREDICT
#PROJECTS=$(find /Shared/johnsonhj/HDNI/20131216_RobotRater/quick_scripts -mindepth 1 -maxdepth 1 -type d ! -name 'HDNI_*')
# TrackOn
PROJECTS=$(find /Shared/johnsonhj/HDNI/20131216_RobotRater/quick_scripts -mindepth 1 -maxdepth 1 -type d -name 'HDNI_*')

# rm insert.log
# touch insert.log

for PROJECT in ${PROJECTS}; do
    echo ${PROJECT}
    SQLFILES=$(find ${PROJECT} -mindepth 3 -maxdepth 3 -type f -name 'batchRated.sql')
    for FILE in ${SQLFILES}; do
        echo "Writing to new file..."
        # HACK
        # NEWFILE=${FILE/%sql/new.sql}
        # tr -s ' ' < ${FILE} >${NEWFILE}
        # psql -d AutoWorkUp -U autoworkup -p 5432 -h psych-db.psychiatry.uiowa.edu -L ${FILE/%sql/log} -f ${NEWFILE} $2>>1 $1>>insert.log
        # END HACK
        psql -d AutoWorkUp -U autoworkup -p 5432 -h psych-db.psychiatry.uiowa.edu -f ${FILE} $2>>1 $1>>insert.log
    done
done
