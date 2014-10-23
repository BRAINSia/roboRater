"""
Usage:
  roboRater -h | --help
  roboRater -V | --version
  roboRater test [-v | --verbose]
  roboRater compare [--force] [-s ID | --session ID] BENCHMARK TESTDIR RESULTDIR REPORT
  roboRater write [--force] (-s ID | --session ID) BENCHMARK TESTDIR RESULTDIR CSVIN SQLOUT

Commands:
  compare           Run comparison
  test              Run doctests

Arguments:
  BENCHMARK         Benchmark experiment directory, i.e. the experiment to compare to
  TESTDIR           Test experiment directory, i.e. the experiment to compare
  RESULTDIR         Result experiment directory
  REPORT            Report file path

Options:
  -h, --help           Show this help and exit
  -V, --version        Print roboRater version
  -v, --verbose        Print more testing information
  --force              Overwrite previous results
  -s ID, --session ID  The session to compare in the form <project>/<subject>/<session>

"""
import csv
import glob
import sys
import os

import SimpleITK as sitk
# assert sitk.Version.MajorVersion() == '0' and sitk.Version.MinorVersion() == '6'
try:
    from AutoWorkup.SEMTools import BRAINSFit
except:
    pass

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


def connectToPostgres(user='autoworkup', database='sinapse_db', testing=False):
    """
    Connect to postgres databases at psych-db.psychiatry.uiowa.edu
    Nota Bene: You must have your password in your keyring

    Inputs:
        user: string, default = 'autoworkup'
        database: string, default = 'imagingmeasurements'
        testing: boolean, default = False

    Outputs: psycopg2._psycopg.connection

    >>> connectToPostgres(testing=True).close()
    >>> connectToPostgres(user='test', testing=True)
    Traceback (most recent call last):
    ...
    AssertionError: Connection not successful!

    """
    connection = None
    password = keyring.get_password('Postgres', user)
    if testing:
        host = 'pre-psych-db.psychiatry.uiowa.edu'
        password = 'ch41rcrush3r'
    else:
        host = 'psych-db.psychiatry.uiowa.edu'
    try:
        connection = sql.connect(host=host, port=5432, database=database,
                                 user=user, password=password)
    except:
        assert isinstance(connection, sql._psycopg.connection), "Connection not successful!"
    return connection


def transform(fixed_path, moving_path, transform_path, image_path, force=False):
    if not force and os.path.exists(transform_path) and os.path.exists(image_path):
        pass
    else:
        print "**** RUNNING BRAINSFit() ****"
        bfit = BRAINSFit()
        inputs = bfit.inputs
        inputs.useRigid = True
        inputs.costMetric = 'MMI'
        inputs.numberOfSamples = 100000
        inputs.fixedVolume = fixed_path
        inputs.movingVolume = moving_path
        inputs.outputVolume = image_path
        inputs.outputTransform = transform_path
        if not os.path.exists(os.path.dirname(transform_path)):
            os.makedirs(os.path.dirname(transform_path))
        print ""
        print bfit.cmdline
        print ""
        result = bfit.run()
        # print "Result is: ", result
        assert os.path.exists(transform_path)
        assert os.path.exists(image_path)
    return sitk.ReadTransform(transform_path)


def resample(moving, fixed, transform=sitk.Transform(), interpolator=sitk.sitkNearestNeighbor, cast=sitk.sitkUInt8):
        m_image = sitk.Cast(moving, cast)
        f_image = sitk.Cast(fixed, cast)
        outImage = f_image
        outImage = sitk.Resample(m_image, f_image, transform, interpolator)
        return outImage


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
          _project,
          _subject
        FROM
          autoworkup_scm.derived_images
        WHERE
          autoworkup_scm.derived_images.location = %s AND
          autoworkup_scm.derived_images._analysis = %s AND
          autoworkup_scm.derived_images.status = 'R' AND
          autoworkup_scm.derived_images._session = %s;
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


def getBenchmarkQA(benchmark, session, labelstr, benchValues):
    isFound = False
    for temp in benchValues:
        if temp['session'] == session and temp['benchmark'] == benchmark:
            isFound = True
            break
    if isFound and temp[labelstr] in ('1', '5'):
        return True
    if not isFound:
        print "No matching benchmark record found"
    return False


def mergeLabels(test, benchmark, labelstr):
    label = LABELS[labelstr]
    threshed = sitk.BinaryThreshold(benchmark,
                                     lowerThreshold=(label - 0.5),
                                     upperThreshold=(label + 0.5),
                                     insideValue=label,
                                     outsideValue=0)
    threshed = sitk.Cast(threshed, sitk.sitkUInt8)
    cleaned = (test != label) * test  # zero out bad label
    cleaned = (threshed != label) * cleaned  # zero out where 'benchmark' will go
    cleaned = threshed + cleaned
    return cleaned


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
    a = benchmark
    b = test
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


def getQAResult(experiment, session, column='*'):
    """
    >>> print getQAResult('l_caudate', '20131119_TrackOn_ManualResults', '218087801_20120907_30')
    <BLANKLINE>
    SELECT autoworkup_scm.image_reviews.l_caudate FROM
        autoworkup_scm.image_reviews NATURAL JOIN
        autoworkup_scm.derived_images
    WHERE
        autoworkup_scm.derived_images._analysis = '20131119_TrackOn_ManualResults' AND
        autoworkup_scm.derived_images._session = '218087801_20120907_30'
    ORDER BY autoworkup_scm.image_reviews.review_time DESC
    LIMIT 1;
    """
    sqlCommand = """
SELECT autoworkup_scm.image_reviews.{column} FROM
    autoworkup_scm.image_reviews NATURAL JOIN
    autoworkup_scm.derived_images
WHERE
    (autoworkup_scm.derived_images."_analysis" = '{experiment}' OR
	 autoworkup_scm.derived_images."_analysis" = substring('{experiment}' FROM \'[0-9]{8}_[A-Za-z]*_\') || \'ManualResults\')
    autoworkup_scm.derived_images._session = '{session}'
ORDER BY autoworkup_scm.image_reviews.review_time DESC
LIMIT 1;""".format(column=column, experiment=experiment, session=session)
    return sqlCommand


def getQARecord(experiment, session, testing=False):
    """
    >>> print getQARecord('20131119_TrackOn_ManualResults', '218087801_20120907_30', testing=True)
    <BLANKLINE>
    SELECT autoworkup_scm.image_reviews.* FROM
        autoworkup_scm.image_reviews NATURAL JOIN
        autoworkup_scm.derived_images
    WHERE
        autoworkup_scm.derived_images._analysis = '20131119_TrackOn_ManualResults' AND
        autoworkup_scm.derived_images._session = '218087801_20120907_30'
    ORDER BY autoworkup_scm.image_reviews.review_time DESC
    LIMIT 1;
    """
    if SQL_ACCESS:
        raise NotImplementedError
        connection = connectToPostgres(testing=testing)
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
        sqlCommand = getQAResult(experiment, session, '.')
        return sqlCommand


def flagForReview():
    query = "SELECT qa_code FROM autoworkup_scm.\"QA_mapping\" WHERE human = 'needs review'"
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
    insert = "INSERT INTO autoworkup_scm.image_reviews ( \
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
    (SELECT reviewer_id FROM autoworkup_scm.reviewers WHERE autoworkup_scm.reviewers.login = 'roborater'), \
    (SELECT record_id FROM autoworkup_scm.derived_images WHERE autoworkup_scm.derived_images._analysis = '{experiment}' AND \
    autoworkup_scm.derived_images._session = '{session}') \
    );".format(experiment=experiment, session=session, **values)
    if SQL_ACCESS:
        raise NotImplementedError
        connection = connectToPostgres(database='AutoWorkUp')
        cursor = connection.cursor()
        try:
            print ""
            print insert
            print ""
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
    sqlCommand = "SELECT autoworkup_scm.image_reviews.review_id FROM \
    autoworkup_scm.image_reviews, autoworkup_scm.derived_images WHERE \
    autoworkup_scm.derived_images.status != 'L' AND autoworkup_scm.derived_images.record_id = autoworkup_scm.image_reviews.record_id AND \
    autoworkup_scm.derived_images._analysis = '{experiment}' AND autoworkup_scm.derived_images._session = '{session}' \
    ORDER BY autoworkup_scm.image_reviews.review_time DESC LIMIT 1;""".format(experiment=experiment, session=session)
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


def roboRater(benchmark_dir, test_dir, session, outpath, outfile, force=False):
    record = tuple(session.split(os.path.sep))
    session = record[-1]
    b_record, b_path, b_T1_path, fixed = createPaths(record, benchmark_dir)
    t_record, t_path, t_T1_path, moving = createPaths(record, test_dir)
    b_experiment = os.path.basename(benchmark_dir)
    t_experiment = os.path.basename(test_dir)
    if SQL_ACCESS:
        if checkForReview(t_experiment, session):
            print "Already reviewed.  Skipping..."
            return 1
        if not checkForReview(b_experiment, session):
            print "No previous review to compare with.  Skipping..."
            return 2

    base = os.path.join(outpath, record[0], record[1], record[2])
    tx_path = os.path.join(base, 'benchmarkToTest.mat')
    image_path = os.path.join(base, 'T1ToTest.nii.gz')
    resample_path = os.path.join(base, 'all_Labels_seg_fixed.nii.gz')

    if not force and os.path.exists(tx_path):
         benchmark2testTx = sitk.ReadImage(tx_path)
    else:
        benchmark2testTx = transform(t_T1_path, b_T1_path, tx_path, image_path)

    if not force and os.path.exists(resample_path):
        resampledImg = sitk.ReadImage(resample_path)
    else:
        resampledImg = resample(moving, fixed, benchmark2testTx, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(resampledImg, resample_path)
    assert os.path.exists(resample_path), "Registered test file must exist!"
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
    compare = compareLabels(resampledImg, testImg)
    values = {}
    for labelImageName in LABELS.keys():
        # stats = getVolumes(registered, compare)
        if qualityThreshold(compare[labelImageName]['dice']):
            # good += 1
            values[labelImageName] = getQAResult(labelImageName, b_experiment, session)
        else:
            #  bad += 1
            values[labelImageName] = flagForReview()
    values = getT1T2TissueLabels(b_experiment, session, values)
    # assert good == GOOD, good
    # assert bad == BAD, bad
    ### END HACK: testing
    results = setQAResult(t_experiment, session, values)
    if not SQL_ACCESS:
        sqlFile = os.path.join(outpath, record[0], record[1], record[2], 'batchRated.sql')
        writeSQL(sqlFile, results)
    else:
        csvFile = os.path.join(outpath, record[0], record[1], record[2], outfile)
        writeCSV(results, csvFile, b_record[0], t_record[0])
    return 0


def roboWriter(session=None, BENCHMARK=None, TESTDIR=None, RESULTDIR=None, CSVIN=None, SQLOUT=None, **args):
    """
    >>> roboWriter('218087801_20120907_30', '/Shared/johnsonhj/TrackOn/Experiments/20131119_TrackOn_ManualResults', '/Shared/johnsonhj/TrackOn/Experiments/20131119_TrackOn_Results', '/tmp/roboWriterTest', 'csvfile', 'sqlfile')
    SELECT roboWrite_fn('20131119_TrackOn_ManualResults', '20131119_TrackOn_Results', '218087801_20120907_30')
    """
    benchValues = {}
    for labelImageName in LABELS.keys():
        benchValues[labelImageName] = getQAResult(labelImageName, b_experiment, session)
    if copyBench:
        assert SQL_ACCESS, "Error: Cannot copy benchmark labels without access to database!"
        # good = ???
        # for label in benchValues.keys():
        #     if bv[label] == good:
        #
    record = tuple(session.split(os.path.sep))
    session = record[-1]
    b_record, b_path, b_T1_path, moving = createPaths(record, benchmark_dir)
    t_record, t_path, t_T1_path, fixed = createPaths(record, test_dir)
    b_experiment = os.path.basename(benchmark_dir)
    t_experiment = os.path.basename(test_dir)
    if SQL_ACCESS:
        if checkForReview(t_experiment, session):
            print "Already reviewed.  Skipping..."
            return 1
        if not checkForReview(b_experiment, session):
            print "No previous review to compare with.  Skipping..."
            return 2

    base = os.path.join(outpath, record[0], record[1], record[2])
    tx_path = os.path.join(base, 'benchmarkToTest.mat')
    image_path = os.path.join(base, 'T1ToTest.nii.gz')
    resample_path = os.path.join(base, 'all_Labels_seg_fixed.nii.gz')
    if not force and os.path.exists(tx_path):
         benchmark2testTx = sitk.ReadImage(tx_path)
    else:
        benchmark2testTx = transform(t_T1_path, b_T1_path, tx_path, image_path, force)
    if not force and os.path.exists(resample_path):
        resampledImg = sitk.ReadImage(resample_path)
    else:
        resampledImg = resample(moving, fixed, benchmark2testTx, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(resampledImg, resample_path)

    testImg = sitk.ReadImage(fixed)
    compare = compareLabels(resampledImg, testImg)
    values = {}
    benchValues = []
    with open(CSVIN, 'rb') as csvin:
        reader = csv.DictReader(csvin, delimiter=',')
        for row in reader:
            benchValues.append(row)
    # Evaluate each label
    labelKeys = LABELS.keys(); labelKeys.sort()
    for labelImageName in labelKeys:
        if qualityThreshold(compare[labelImageName]['dice']):
            values[labelImageName] = getQAResult(labelImageName, b_experiment, session)
        elif getBenchmarkQA(b_experiment, session, labelImageName, benchValues):
            # region in test is 'bad' compared to benchmark AND region in benchmark is 'good'
            if 'linearImg' not in locals():
                linearImg = resample(moving, fixed, transform, interpolator=sitk.sitkLinear, cast=sitk.sitkFloat32)
            linearImg = sitk.Cast(mergeLabels(fixedImg, linearImg, labelImageName), sitk.sitkFloat32)
            values[labelImageName] = 'SELECT qa_code FROM autoworkup_scm."QA_mapping" WHERE human = "roboWriter-acceptable"'
        else:
            values[labelImageName] = flagForReview()
    # Write final label image
    if 'linearImg' in locals():
        stripcount = len('.nii.gz')
        clean_fname = os.path.basename(moving)[:-stripcount] + '_roboRaterCleaned.nii.gz'
        registered_path = os.path.join(os.path.dirname(transform_path), clean_fname)
        sitk.WriteImage(sitk.Cast(linearImg, sitk.sitkUInt8), registered_path)
    else:
        registered_path = os.path.join(os.path.dirname(transform_path), os.path.basename(moving))
        sitk.WriteImage(resampledImg, registered_path)
    assert os.path.exists(registered_path), "Registered test file must exist!"
    # Create the image review record
    values = getT1T2TissueLabels(b_experiment, session, values)
    results = setQAResult(t_experiment, session, values)
    if not SQL_ACCESS:
        sqlFile = os.path.join(outdir, record[0], record[1], record[2], 'batchRated.sql')
        writeSQL(sqlFile, results)
    else:
        csvFile = os.path.join(outdir, record[0], record[1], record[2], outfile)
        writeCSV(results, csvFile, b_record[0], t_record[0])
    return 0


if __name__ == "__main__":
    import sys

    from docopt import docopt

    args = docopt(__doc__)
    if args['test']:
        import doctest

        # SQL_ACCESS = False
        # doctest.testmod(verbose=True)
        SQL_ACCESS = True
        import psycopg2 as sql
        import keyring

        doctest.testmod(verbose=args['--verbose'])
    elif args['compare']:
        reval = roboRater(args['BENCHMARK'], args['TESTDIR'], args['--session'], args['RESULTDIR'], args['REPORT'], args['--force'])
        sys.exit(retval)
    elif args['write']:
        inputs = {}
        for key in args.keys():
            inputs[key.strip('-')] = args.pop(key)
        print roboWriter(**inputs)
        sys.exit(retval)
