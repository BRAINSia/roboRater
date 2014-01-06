"""
Usage:
  roboRater [-b | --benchmark] DIR1 [-t | --test] DIR2 [-s | --session SESSION] [-o | --outdir] OUTDIR [-r | --report] FILE
  roboRater -h | --help
  roboRater -v | --version

Options:
  -h, --help                     Show this help and exit
  -v, --version                  Print roboRater version
  -b DIR1, --benchmark DIR1      The benchmark experiment directory, i.e. the experiment to compare to
  -t DIR2, --test DIR2           The test experiment directory, i.e. the experiment to compare
  -s SESSION, --session SESSION  The session to compare
  -o OUTDIR, --outdir OUTDIR     Result output directory
  -r FILE, --report FILE         The output report file
"""
# -*- coding: utf-8 -*-
# https://www.icts.uiowa.edu/jira/browse/PREDICTIMG-3082
#
# Description
# ===========
#
# Create an application that compares segmentations from one experiment to another and autopopulates the QA review if the segmentations are sufficiently similar.
#
# Requirements
# ============
#
# * query/insert records in the AutoWorkUp database
# * writes out a report with useful stats/lists of the comparison results
# * command line:
#
#     `roboRater --benchmark /path/to/benchmark/experiment/directory/ \
#     --test /path/to/experiment/to/be/compared/directory/ \
#     --report /path/to/report/file`
#
# **NOTICE**
#
# Must run this code in an environment with a working SEMTools package, i.e.
#
#     source /paulsen/Experiments/rsFMRI/rs-fMRI-pilot/ENV/bin/activate
#     ipython notebook
#
# __Steps:__
#
# 1. SELECT all sessions where benchmark was performed
# 1. for session in sessions:
#     1. If files missing, add to missing_labels list
#     1. Register the T1-average from test to benchmark
#     1. Apply transform to test labels
#     1. Compare using DICE and others
#     1, If heuristics >= some value:
#         1. INSERT corresponding benchmark QA values into test QA entry with user='roboRater'
#         1. Add to successful count
#     1. else:
#         1. Add file path to failed_comparison list
# 1. Write out report with stats, missing_labels, and failed_comparison lists
#
import csv
import glob
import sys
import os

import SimpleITK as sitk

from SEMTools import BRAINSFit

# globals
DICE_THRESHOLD = 0.9
SQL_ACCESS = False
if SQL_ACCESS:
    import psycopg2 as sql
    import keyring

LABELS = {"caudate_left": 1,
          "caudate_right": 2,
          "putamen_left": 3,
          "putamen_right": 4,
          "hippocampus_left": 5,
          "hippocampus_right": 6,
          "thalamus_left": 7,
          "thalamus_right": 8,
          "accumben_left": 9,
          "accumben_right": 10,
          "globus_left": 11,
          "globus_right": 12}
### HACK: testing
# good = 0
# bad = 0
# GOOD = 15 # 0 # 6 # 7
# BAD =  0 # 15 # 9 # 8
### END HACK

def connectToPostgres(user='autoworkup', database='imagingmeasurements'):
    """
    Connect to postgres databases at psych-db.psychiatry.uiowa.edu
    Nota Bene: You must have your password in your keyring

    Inputs:
        user: string, default = 'autoworkup'
        database: string, default = 'imagingmeasurements'

    Outputs: psycopg2.connection
    """
    password = keyring.get_password('Postgres', user)
    connection = sql.connect(host='psych-db.psychiatry.uiowa.edu', port=5432, database=database,
                             user=user, password=password)
    return connection

def transform(record, b_T1_path, t_T1_path, outpath, force=False):
    deformedImageName = os.path.join(outpath, record[0], record[1], record[2], 'T1ToTest.nii.gz')
    transform_path = os.path.join(outpath, record[0], record[1], record[2], 'benchmarkToTest.mat')
    if not force and os.path.exists(transform_path) and os.path.exists(deformedImageName):
        return transform_path
    else:
        print "**** RUNNING BRAINSFit() ****"
        bfit = BRAINSFit()
        inputs = bfit.inputs
        inputs.useRigid = True
        inputs.costMetric = 'MMI'
        inputs.numberOfSamples = 100000
        inputs.fixedVolume = b_T1_path
        inputs.movingVolume = t_T1_path
        inputs.outputVolume = deformedImageName
        inputs.outputTransform = transform_path
        if not os.path.exists(os.path.dirname(transform_path)):
            os.makedirs(os.path.dirname(transform_path))
        print ""
        print bfit.cmdline
        print ""
        result = bfit.run()
        print "Result is: ", result
        assert os.path.exists(transform_path)
        assert os.path.exists(deformedImageName)
    return transform_path


def resample(moving, fixed, outfile, tfile, interpolator, force=False):
    if not force and os.path.exists(outfile):
        return 0
    else:
        m_image = sitk.ReadImage(moving)
        f_image = sitk.ReadImage(fixed)
        mytransform = sitk.ReadTransform(tfile)
        outImage = f_image
        outImage = sitk.Resample(m_image, f_image, mytransform, interpolator)
        sitk.WriteImage(outImage, outfile)
    return 0


def find_record(benchmark_dir, session):
    assert os.path.isdir(benchmark_dir) and os.path.exists(benchmark_dir)
    (bmark_base, benchmark_experiment) = os.path.split(benchmark_dir.rstrip(os.path.sep))
    path = glob.glob(os.path.join(benchmark_dir, '*', '*', session))
    assert len(path) == 1, "Two directories with the same session were found. This should never happen!"
    path, _session = os.path.split(path[0])
    path, subject = os.path.split(path)
    path, project = os.path.split(path)
    if SQL_ACCESS:
        raise NotImplementedError
        benchmark_query = """SELECT
          derived_images._project,
          derived_images._subject
        FROM
          public.derived_images
        WHERE
          derived_images.location = %s AND
          derived_images._analysis = %s AND
          derived_images.status = 'R' AND
          derived_images._session = %s;
        """
        connection = connectToPostgres(database='AutoWorkUp')
        cursor = connection.cursor()
        try:
            cursor.execute(benchmark_query, (bmark_base, benchmark_experiment, session))
            benchmark_records = cursor.fetchone()  # There will only be one!
        except sql.Error as err:
            print err.pgcode, ':', err.pgerror
        finally:
            cursor.close()
            connection.close()
    return (project, subject, session)


def createPaths(record, experiment_dir):
    _record = (experiment_dir, ) + record
    path = os.path.join(*list(_record))
    if not os.path.isdir(path) or not os.path.exists(path):
        os.makedirs(path)
    assert os.path.isdir(path) and os.path.exists(path), path
    t1_path = os.path.join(path, 'TissueClassify', 't1_average_BRAINSABC.nii.gz')
    assert os.path.exists(t1_path), "File not found: {0}".format(t1_path)
    labels_path = os.path.join(path, 'CleanedDenoisedRFSegmentations', 'allLabels_seg.nii.gz')
    assert os.path.exists(labels_path), "File not found: {0}".format(labels_path)
    return _record, path, t1_path, labels_path


def getVolumes(labelImageName, compare):
    labelImage = sitk.ReadImage(labelImageName)
    lstat = sitk.LabelStatisticsImageFilter()
    lstat.Execute(labelImage, labelImage)
    ImageSpacing = labelImage.GetSpacing()
    for name in compare.keys():
        value = LABELS[name]
        if lstat.HasLabel(value):
            myMeasurementMap = lstat.GetMeasurementMap(value)
            dictKeys = myMeasurementMap.GetVectorOfMeasurementNames()
            dictValues = myMeasurementMap.GetVectorOfMeasurementValues()
            measurementDict = dict(zip(dictKeys, dictValues))
            compare[name]['volume'] = ImageSpacing[0] * ImageSpacing[1] * ImageSpacing[2] * measurementDict['Count']
    return compare


def writeCSV(compare, csvfile, baseline, test):
    # [Struct, vol, hausdorff, hausdorffAvg, dice, border, baseline, test]
    with open(csvfile, 'w') as fid:
        writer = csv.DictWriter(fid, fieldnames=['structure', 'volume', 'hausdorff', 'hausdorffAvg', 'dice', 'border', 'benchmark', 'test'])
        writer.writeheader()
        for key in compare.keys():  # Struct
            temp = compare[key]
            temp['structure'] = key
            temp['benchmark'] = baseline
            temp['test'] = test
            writer.writerow(temp)


def compareLabels(benchmark, test):
    def dice(a, b):
        d = 2 * sitk.Statistics(a & b)["Sum"] / sitk.Statistics(a + b)["Sum"]
        return d

    def hamming_distance(a, b):
        "Return the Hamming distance between equal-length sequences."
        if len(a) != len(b):
            return levenshtein_dist(a, b)  # raise ValueError("Undefined for sequences of unequal length")
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def levenshtein_dist(a, b):
        assert len(a) != 0 and len(b) != 0
        if a[len(a)-1] == b[len(b)-1]:
            cost = 0
        else:
            cost = 1

    def border(a, b):
        "Compare the borders of a and b"
        edge_a = a ^ sitk.BinaryErode(a)
        edge_b = b ^ sitk.BinaryErode(b)
        return dice(edge_a, edge_b)

    def hausdorff(a, b):
        out = {}
        h = sitk.HausdorffDistanceImageFilter()
        h.Execute(a, b)
        out['hausdorff'] = h.GetHausdorffDistance()
        out['hausdorffAvg'] = h.GetAverageHausdorffDistance()
        return out

    # MAIN compareLabels
    a = sitk.ReadImage(benchmark)
    b = sitk.ReadImage(test)
    retval = {}
    for key, value in LABELS.items():
        a_label = (a == value)
        b_label = (b == value)
        result = {}
        # result = hausdorff(a_label, b_label)
        # result['border'] = border(a_label, b_label)
        # result['structure'] = key
        result['dice'] = dice(a_label, b_label)
        retval[key] = result
    return retval


def qualityThreshold(d):
    if d >= DICE_THRESHOLD:
        return True
    else:
        return False


def getQAResult(column, experiment, session):
    sqlCommand = "SELECT image_reviews.{column} FROM \
    public.image_reviews, public.derived_images WHERE \
    derived_images.record_id = image_reviews.record_id AND \
    derived_images._analysis = '{experiment}' AND \
    derived_images._session = '{session}' \
    ORDER BY image_reviews.review_time DESC \
    LIMIT 1".format(column=column, experiment=experiment, session=session)
    if SQL_ACCESS:
        raise NotImplementedError
        connection = connectToPostgres(database='AutoWorkUp')
        cursor = connection.cursor()
        try:
            cursor.execute(sqlCommand)
            benchmark_record = cursor.fetchone()
            return benchmark_record
        except sql.Error as err:
            print err.pgcode, ':', err.pgerror
        finally:
            cursor.close()
            connection.close()
    else:
        return sqlCommand


def flagForReview():
    query = "SELECT qa_code FROM \"QA_mapping\" WHERE human = 'needs review'"
    if SQL_ACCESS:
        raise NotImplementedError
    else:
        return query


def getT1T2TissueLabels(experiment, session, values):
    """ If caudates and putamens are consistent AND good, copy T1/T2/Tissue labels QA """
    caudate_putamen = [values[x] for x in ('caudate_right', 'caudate_left', 'putamen_right', 'putamen_left')]
    columns = ('t1_average', 'labels_tissue', 't2_average')
    bad_value = flagForReview()
    ### HACK: testing
    # global bad, good
    for column in columns:
        if bad_value in caudate_putamen:
    #         bad += 1
            values[column] = bad_value
        else:
    #         good += 1
    ### END HACK
            values[column] = getQAResult(column, experiment, session)
    return values


def setQAResult(experiment, session, values):
    insert = "INSERT INTO image_reviews ( \
    accumben_right, accumben_left, \
    caudate_right, caudate_left, \
    globus_right, globus_left, \
    hippocampus_right, hippocampus_left, \
    putamen_right, putamen_left, \
    thalamus_right, thalamus_left, \
    t1_average, labels_tissue, t2_average, \
    reviewer_id, record_id) \
    VALUES ( \
    ({accumben_right}), ({accumben_left}), \
    ({caudate_right}), ({caudate_left}), \
    ({globus_right}), ({globus_left}), \
    ({hippocampus_right}), ({hippocampus_left}), \
    ({putamen_right}), ({putamen_left}), \
    ({thalamus_right}), ({thalamus_left}), \
    ({t1_average}), ({labels_tissue}), ({t2_average}), \
    (SELECT reviewer_id FROM  public.reviewers WHERE reviewers.login = 'roborater'), \
    (SELECT record_id FROM public.derived_images WHERE derived_images._analysis = '{experiment}' AND \
    derived_images._session = '{session}') \
    );".format(experiment=experiment, session=session, **values)
    if SQL_ACCESS:
        raise NotImplementedError
        connection = connectToPostgres(database='AutoWorkUp')
        cursor = connection.cursor()
        try:
            cursor.execute(insert)
        except sql.Error as err:
            print err.pgcode, ':', err.pgerror
        finally:
            cursor.close()
            connection.close()
        return 0
    else:
        return insert


def checkForReview(experiment, session):
    sqlCommand = "SELECT image_reviews.review_id FROM \
    public.image_reviews, public.derived_images WHERE \
    derived_images.status != 'L' AND derived_images.record_id = image_reviews.record_id AND \
    derived_images._analysis = '{experiment}' AND derived_images._session = '{session}' \
    ORDER BY image_reviews.review_time DESC LIMIT 1;""".format(experiment=experiment, session=session)
    connection = connectToPostgres(database='AutoWorkUp')
    cursor = connection.cursor()
    try:
        print sqlCommand
        cursor.execute(sqlCommand)
        benchmark_record = cursor.fetchone()
    except sql.Error as err:
        print err.pgcode, ':', err.pgerror
    finally:
        cursor.close()
        connection.close()
    if not benchmark_record:
        return False
    return True


def writeSQL(filename, sql_cmd):
    with open(filename, 'w') as fid:
        fid.write(sql_cmd + '\n')
        fid.flush()


def roboRater(benchmark_dir, test_dir, session, outdir, outfile, force=False):
    record = tuple(session.split(os.path.sep))
    session = record[-1]
    b_record, b_path, b_T1_path, fixed = createPaths(record, benchmark_dir)
    t_record, t_path, t_T1_path, moving = createPaths(record, test_dir)
    b_experiment = os.path.basename(benchmark_dir)
    t_experiment = os.path.basename(test_dir)
    if SQL_ACCESS and checkForReview(t_experiment, session):
        print "Already reviewed.  Skipping..."
        return 0
    if SQL_ACCESS and not checkForReview(b_experiment, session):
        print "No previous review to compare with.  Skipping..."
        return 0
    transform_path = transform(record, b_T1_path, t_T1_path, outdir, force)
    registered = os.path.join(os.path.dirname(transform_path), os.path.basename(moving))
    resample(moving, fixed, registered, transform_path, interpolator=sitk.sitkNearestNeighbor, force=force)
    assert os.path.exists(registered), "Registered test file must exist!"
    compare = compareLabels(fixed, registered)
    ### HACK: testing
    # global bad, good
    # compare = {}
    # for label in LABELS.keys():
    #     import sys, re
    #     # test = re.search('left', label)
    #     # test = re.search('putamen|caudate', label)
    #     # test = None
    #     test = '1'
    #     if not test is None:
    #         compare[label] = {'dice': 0.90}  # ALL SHOULD PASS
    #     else:
    #         print label
    #         compare[label] = {'dice': 0.90 - sys.float_info.epsilon}  # ALL SHOULD FAIL
    values = {}
    for labelImageName in LABELS.keys():
        # stats = getVolumes(registered, compare)
        if qualityThreshold(compare[labelImageName]['dice']):
    #         good += 1
            values[labelImageName] = getQAResult(labelImageName, b_experiment, session)
        else:
    #         bad += 1
            values[labelImageName] = flagForReview()
    values = getT1T2TissueLabels(b_experiment, session, values)
    # assert good == GOOD, good
    # assert bad == BAD, bad
    ### END HACK: testing
    results = setQAResult(t_experiment, session, values)
    if not SQL_ACCESS:
        sqlFile = os.path.join(outdir, record[0], record[1], record[2], 'batchRated.sql')
        writeSQL(sqlFile, results)
    else:
        csvFile = os.path.join(outdir, record[0], record[1], record[2], outfile)
        writeCSV(results, csvFile, b_record[0], t_record[0])
    return 0


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='roboRater')
    parser.add_argument('-b', '--benchmark', action='store', dest='benchmark_dir', type=str, required=True,
                        help='The benchmark experiment directory, i.e. the experiment to compare to')
    parser.add_argument('-t', '--test', action='store', dest='test_dir', type=str, required=True,
                        help='The test experiment directory, i.e. the experiment to compare')
    parser.add_argument('-s','--session', action='store', dest='session', type=str, required=True,
                        help='Unique path to session, e.g. "site/subject/session"')
    parser.add_argument('-o', '--outdir', action='store', dest='outdir', type=str, required=True,
                        help='Result output directory')
    parser.add_argument('-r','--report', action='store', dest='outfile', type=str, required=True,
                        help='The filename for the output csv.  If it exists already, it will be appended.')
    parser.add_argument('-f', '--force', action='store_true', dest='force_registration', required=False,
                        help="Force registration if registered files already exist")
    args = parser.parse_args()
    sys.exit(roboRater(args.benchmark_dir, args.test_dir, args.session, args.outdir, args.outfile, args.force_registration))
