"""
Usage:
  roboRater -h | --help
  roboRater -V | --version
  roboRater test [-v | --verbose]
  roboRater compare [--force] CONFIG TEST REPORT
  roboRater write   [--force] TEST BENCHMARK RESULTS CSVIN SQLOUT

Commands:
  compare           Run comparison
  test              Run doctests
  write             Run comparison and create merged segmentations based on results

Arguments:
  CONFIG            Configuration file for inputs/outputs
  BENCHMARK         Benchmark experiment directory, i.e. the experiment to compare to
  TEST              Test experiment directory (with site/subject/session), i.e. the experiment to compare
  RESULTS           Result experiment directory
  REPORT            Report file path
  CSVIN             Comma-separated file path with headings
  SQLOUT            SQL-generated output file name

Options:
  -h, --help           Show this help and exit
  -V, --version        Print roboRater version
  -v, --verbose        Print more testing information
  --force              Overwrite previous results

"""
import csv
import glob
import sys
import os
import warnings

import SimpleITK as sitk
# assert sitk.Version.MajorVersion() == '0' and sitk.Version.MinorVersion() == '6'
try:
    from AutoWorkup.SEMTools import BRAINSFit
except:
    warnings.warn("Cannot find Python wrapping for BRAINSFit", RuntimeWarning)
try:
    import sinapse.path
except:
    raise

# globals
DICE_THRESHOLD = 0.9
SQL_ACCESS = False
if SQL_ACCESS:
    import psycopg2 as sql
    import keyring

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

    # >>> connectToPostgres(testing=True).close()
    # >>> connectToPostgres(user='test', testing=True)
    # Traceback (most recent call last):
    # ...
    # AssertionError: Connection not successful!

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
    """
    >>> moving = os.path.join(os.path.dirname(__file__), 'tests/benchmark.nii.gz')
    >>> original = os.path.join(os.path.dirname(__file__), 'tests/target.nii.gz')
    >>> fixed = os.path.join(os.path.dirname(__file__), 'tests/test.nii.gz')
    >>> b2t_tx = sitk.ReadTransform(os.path.join(os.path.dirname(__file__), 'tests/benchmark2testTx.mat'))
    >>> try:
    ...     assert b2t_tx == transform(fixed, moving, 'transform_doctest.mat', 'transform_doctest.nii.gz')
    ...     compare = original - sitk.ReadImage('transform_doctest.nii.gz')
    ...     array = sitk.GetArrayFromImage(compare)
    ...     assert len(array.nonzero()[0]) < (original.GetSize()[0] * original.GetSize()[1] * 0.0005)
    ... except:
    ...     raise
    """
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
    """
    >>> try:
    ...     moving = sitk.ReadImage(os.path.join(os.path.dirname(__file__), 'tests/benchmark.nii.gz'))
    ...     original = sitk.ReadImage(os.path.join(os.path.dirname(__file__), 'tests/target.nii.gz'))
    ...     fixed = sitk.ReadImage(os.path.join(os.path.dirname(__file__), 'tests/test.nii.gz'))
    ...     b2t_tx = sitk.ReadTransform(os.path.join(os.path.dirname(__file__), 'tests/benchmark2testTx.mat'))
    ...     compare = original - resample(moving, fixed, b2t_tx)
    ...     sitk.WriteImage(compare, os.path.join(os.path.dirname(__file__), 'tests/output/resample_doctest.nii.gz'))
    ...     array = sitk.GetArrayFromImage(compare)
    ...     assert len(array.nonzero()[0]) < (original.GetSize()[0] * original.GetSize()[1] * 0.0005)
    ... except:
    ...     raise
    """
    raise DeprecationWarning
    return 0
    try:
        outImage = sitk.Resample(moving, fixed, transform, interpolator, 0.0, cast)
    except NotImplementedError:
        print interpolator, sitk.GetPixelIDValueAsString(cast)
        raise
    return outImage


def find_files(path, t1_suffix, labels_suffix):
    """
    >>> find_files('/Shared/paulsen/Experiments/20131124_PREDICTHD_Results/PHD_024/0029/34504', 'TissueClassify/t1_average_BRAINSABC.nii.gz', 'CleanedDenoisedRFSegmentations/allLabels_seg.nii.gz')
    ('/Shared/paulsen/Experiments/20131124_PREDICTHD_Results/PHD_024/0029/34504/TissueClassify/t1_average_BRAINSABC.nii.gz', '/Shared/paulsen/Experiments/20131124_PREDICTHD_Results/PHD_024/0029/34504/CleanedDenoisedRFSegmentations/allLabels_seg.nii.gz')
    """
    # if not os.path.isdir(path) or not os.path.exists(path):
    #     os.makedirs(path)
    assert os.path.isdir(path) and os.path.exists(path), path
    t1_path = os.path.join(path, t1_suffix)
    assert os.path.exists(t1_path), "File not found: {0}".format(t1_path)
    labels_path = os.path.join(path, labels_suffix)
    assert os.path.exists(labels_path), "File not found: {0}".format(labels_path)
    return t1_path, labels_path


def get_benchmark_quality(benchmark, session, labelstr, benchValues):
    """
    >>> get_benchmark_quality('20131124_PREDICTHD_Results', '34504', 'caudate_left', [{'session':'34504', 'benchmark':'20131124_PREDICTHD_Results', 'caudate_left':'-1'}])
    False
    >>> get_benchmark_quality('20131124_PREDICTHD_Results', '99999', 'caudate_left', [{'session':'34504', 'benchmark':'20131124_PREDICTHD_Results', 'caudate_left':'1'}])
    No matching benchmark record found
    False

    >>> get_benchmark_quality('19000101_PREDICTHD_Results', '34504', 'caudate_left', [{'session':'34504', 'benchmark':'20131124_PREDICTHD_Results', 'caudate_left':'1'}])
    No matching benchmark record found
    False

    >>> get_benchmark_quality('20131124_PREDICTHD_Results', '34504', 'caudate_left', [{'session':'34504', 'benchmark':'20131124_PREDICTHD_Results', 'caudate_left':'1'}])
    True

    >>> get_benchmark_quality('20131124_PREDICTHD_Results', '34504', 'caudate_left', [{'session':'34504', 'benchmark':'20131124_PREDICTHD_Results', 'caudate_left':'5'}])
    True

    """
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


def modify_labels(target, test, labelstr):
    """
    >>> target = sitk.Cast(sitk.ReadImage(os.path.join(os.path.dirname(__file__), 'tests/target.nii.gz')), sitk.sitkFloat32)
    >>> test = sitk.ReadImage(os.path.join(os.path.dirname(__file__), 'tests/test.nii.gz'))
    >>> count = 1
    >>> for label in LABELS.keys():
    ...     if LABELS[label] > 7:
    ...         continue
    ...     test = modify_labels(target, test, label)
    ...     sitk.WriteImage(test, 'tests/output/modify_labels_%s_doctest_%s.nii.gz' % (label, count))
    ...     count += 1
    >>> try:
    ...     compare = test - sitk.Cast(target, sitk.sitkUInt8)
    ...     array = sitk.GetArrayFromImage(compare)
    ...     assert len(array.nonzero()[0]) < (target.GetSize()[0] * target.GetSize()[1] * 0.0005)
    ... except:
    ...     raise
    >>>
    """
    label = LABELS[labelstr]
    threshed = sitk.BinaryThreshold(target,
                                    lowerThreshold=(label - 0.5),
                                    upperThreshold=(label + 0.5),
                                    insideValue=label,
                                    outsideValue=0)
    # threshed = sitk.Cast(threshed, sitk.sitkUInt8)
    cleaned = (test != label) * test  # zero out bad label
    cleaned = (threshed != label) * cleaned  # zero out where 'target' will go
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


def createLabelMapping(config):
    retval = {}
    for section in config.sections():
        if config.has_option(section, 'benchmark'):
            if config.has_option(section, 'test'):
                retval[section] = (config.getint(section, 'benchmark'), config.getint(section, 'test'))
            else:
                retval[section] = (config.getint(section, 'benchmark'),) * 2
    return retval


def compare_labels(benchmark, test, labelMap):
    """
    >>> compare_labels(sitk.ReadImage('tests/target.nii.gz'), sitk.ReadImage('tests/benchmark.nii.gz'))
    {'globus_right': {'dice': 0.0}, 'caudate_right': {'dice': 0.0}, 'thalamus_right': {'dice': 0.0}, 'globus_left': {'dice': 0.0}, 'hippocampus_left': {'dice': 0.054785020804438284}, 'putamen_left': {'dice': 0.0}, 'accumben_left': {'dice': 0.0}, 'hippocampus_right': {'dice': 0.0}, 'thalamus_left': {'dice': 0.0}, 'putamen_right': {'dice': 0.38096171756286457}, 'caudate_left': {'dice': 0.0639269406392694}, 'accumben_right': {'dice': 0.0}}

    >>> compare_labels(sitk.ReadImage('tests/target.nii.gz'), sitk.ReadImage('tests/target.nii.gz'))
    {'globus_right': {'dice': 0.0}, 'caudate_right': {'dice': 1.0}, 'thalamus_right': {'dice': 0.0}, 'globus_left': {'dice': 0.0}, 'hippocampus_left': {'dice': 1.0}, 'putamen_left': {'dice': 1.0}, 'accumben_left': {'dice': 0.0}, 'hippocampus_right': {'dice': 1.0}, 'thalamus_left': {'dice': 1.0}, 'putamen_right': {'dice': 1.0}, 'caudate_left': {'dice': 1.0}, 'accumben_right': {'dice': 0.0}}

    """
    def dice(a, b):
        try:
            d = 2 * sitk.Statistics(a & b)["Sum"] / sitk.Statistics(a + b)["Sum"]
        except ZeroDivisionError:  # labels don't overlap
            d = 0.0
        except:
            d = None
            raise
        finally:
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

    # MAIN compare_labels
    a = benchmark
    b = test
    retval = {}
    for key, (alabel, blabel) in labelMap.items():
        a_label = (a == alabel)
        b_label = (b == blabel)
        result = {}
        # result = hausdorff(a_label, b_label)
        # result['border'] = border(a_label, b_label)
        # result['structure'] = key
        result['dice'] = dice(a_label, b_label)
        retval[key] = result
    return retval


def getQAResult(experiment, session, column='*'):
    """
    >>> print getQAResult('20131119_TrackOn_ManualResults', '218087801_20120907_30', 'caudate_left')  #doctest: +REPORT_NDIFF, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SELECT DISTINCT ON (autoworkup_scm.derived_images._session)
        autoworkup_scm.image_reviews.caudate_left
    FROM
        autoworkup_scm.image_reviews NATURAL JOIN
        autoworkup_scm.derived_images
    WHERE
        (autoworkup_scm.derived_images._analysis = '20131119_TrackOn_ManualResults' OR
    	 autoworkup_scm.derived_images._analysis = substring('20131119_TrackOn_ManualResults' FROM '20[0-3][0-9][0-1][0-9]_[A-Za-z]*_') || 'ManualResults') AND
        autoworkup_scm.derived_images._session = '218087801_20120907_30'
    ORDER BY autoworkup_scm.derived_images._session, autoworkup_scm.image_reviews.review_time DESC
    """
    sqlCommand = """
SELECT DISTINCT ON (autoworkup_scm.derived_images._session)
    autoworkup_scm.image_reviews.{region}
FROM
    autoworkup_scm.image_reviews NATURAL JOIN
    autoworkup_scm.derived_images
WHERE
    (autoworkup_scm.derived_images._analysis = '{experiment}' OR
	 autoworkup_scm.derived_images._analysis = substring('{experiment}' FROM '20[0-3][0-9][0-1][0-9]_[A-Za-z]*_') || 'ManualResults') AND
    autoworkup_scm.derived_images._session = '{session}'
ORDER BY autoworkup_scm.derived_images._session, autoworkup_scm.image_reviews.review_time DESC""".format(region=column,
                                                                                                         experiment=experiment,
                                                                                                          session=session)
    return sqlCommand


def getQARecord(experiment, session):
    """
    >>> print getQARecord('20131119_TrackOn_ManualResults', '218087801_20120907_30')  #doctest: +REPORT_NDIFF, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SELECT DISTINCT ON (autoworkup_scm.derived_images._session)
        autoworkup_scm.image_reviews.*
    FROM
        autoworkup_scm.image_reviews NATURAL JOIN
        autoworkup_scm.derived_images
    WHERE
        (autoworkup_scm.derived_images._analysis = '20131119_TrackOn_ManualResults' OR
    	 autoworkup_scm.derived_images._analysis = substring('20131119_TrackOn_ManualResults' FROM '20[0-3][0-9][0-1][0-9]_[A-Za-z]*_') || 'ManualResults') AND
        autoworkup_scm.derived_images._session = '218087801_20120907_30'
    ORDER BY autoworkup_scm.derived_images._session, autoworkup_scm.image_reviews.review_time DESC
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
        sqlCommand = getQAResult(experiment, session)
        return sqlCommand


def flagForReview():
    query = "SELECT qa_code FROM autoworkup_scm.\"QA_mapping\" WHERE human = 'needs review'"
    if SQL_ACCESS:
        raise NotImplementedError
    else:
        return query


def getT1T2TissueLabels(experiment, session, values):
    """ If caudates and putamens are consistent AND good, copy T1/T2/Tissue labels QA

    >>> try:
    ...     values = getT1T2TissueLabels('20131119_TrackOn_ManualResults', '218087801_20120907_30', {'caudate_right':flagForReview(), 'caudate_left':'SELECT ...', 'putamen_right':'SELECT ...', 'putamen_left':'SELECT ...'})
    ...     assert values['t1_average'] == flagForReview()
    ... except:
    ...     raise

    >>> try:
    ...     values = getT1T2TissueLabels('20131119_TrackOn_ManualResults', '218087801_20120907_30', {'caudate_right':'SELECT ...', 'caudate_left':'SELECT ...', 'putamen_right':'SELECT ...', 'putamen_left':'SELECT ...'})
    ...     assert values['t1_average'] != flagForReview()
    ... except:
    ...     raise

    """
    caudate_putamen = [values[x] for x in ('caudate_right', 'caudate_left', 'putamen_right', 'putamen_left')]
    columns = ('t1_average', 'labels_tissue', 't2_average')
    bad_value = flagForReview()
    for column in columns:
        if bad_value in caudate_putamen:
            values[column] = bad_value
        else:
            values[column] = getQAResult(experiment, session, column)
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
    if SQL_ACCESS:
        raise NotImplementedError
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


def common(config, TEST, force=False):
    BENCHMARK = config.get('benchmark', 'base')
    _junk, project, subject, session = sinapse.path.parse(TEST)
    b_session = os.path.join(BENCHMARK, project, subject, session)
    try:
        b_T1_path, moving = find_files(b_session, config.get('benchmark', 't1'), config.get('benchmark', 'labels'))
    except AssertionError:
        raise
    try:
        t_T1_path, fixed = find_files(TEST, config.get('test', 't1'), config.get('test', 'labels'))
    except AssertionError:
        raise
    b_experiment = os.path.basename(BENCHMARK)
    t_experiment = os.path.basename(TEST)
    if SQL_ACCESS:
        if checkForReview(t_experiment, session):
            print "Already reviewed.  Skipping..."
            sys.exit(1)
        if not checkForReview(b_experiment, session):
            print "No previous review to compare with.  Skipping..."
            sys.exit(2)
    return b_T1_path, sitk.ReadImage(moving), t_T1_path, sitk.ReadImage(fixed), project, subject, session


def roboRater(config, TEST, REPORT, force=False, **args):
    b_T1_path, moving, t_T1_path, fixed, project, subject, session = common(config, TEST, force)
    base = os.path.join(config.get('results', 'base'), project, subject, session, config.get('results', 'folder'))
    tx_path = os.path.join(base, config.get('results', 'transform'))
    image_path = os.path.join(base, config.get('results', 't1'))
    resample_path = os.path.join(base, config.get('results', 'labels'))
    if not force and os.path.exists(tx_path):
         benchmark2testTx = sitk.ReadTransform(tx_path)
    else:
        benchmark2testTx = transform(t_T1_path, b_T1_path, tx_path, image_path, force)
    if not force and os.path.exists(resample_path):
        resampledImg = sitk.ReadImage(resample_path)
    else:
        resampledImg = sitk.Resample(moving, fixed, benchmark2testTx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
        sitk.WriteImage(resampledImg, resample_path)
    labelMap = createLabelMapping(config)
    compare = compare_labels(resampledImg, fixed, labelMap)
    values = {}
    for labelImageName in labelMap.keys():
        if compare[labelImageName]['dice'] >= DICE_THRESHOLD:
            values[labelImageName] = getQAResult(labelImageName, b_experiment, session)
        else:
            values[labelImageName] = flagForReview()
    values = getT1T2TissueLabels(os.path.basename(config.get('benchmark', 'base')), session, values)
    results = setQAResult(os.path.basename(config.get('test', 'base')), session, values)
    if not SQL_ACCESS:
        sqlFile = os.path.join(config.get('results', 'base'), project, subject, session, REPORT)
        writeSQL(sqlFile, results)
    else:
        csvFile = os.path.join(outpath, project, subject, session, REPORT)
        writeCSV(results, csvFile, b_record[0], t_record[0])
    return 0


def roboWriter(record, BENCHMARK, TEST, RESULTS, CSVIN=None, SQLOUT=None, force=False, **args):
    """
    >>> roboWriter('HDNI_002/218087801/218087801_20120907_30', '/Shared/johnsonhj/TrackOn/Experiments/20131119_TrackOn_ManualResults', '/Shared/johnsonhj/TrackOn/Experiments/20131119_TrackOn_Results', '/tmp/roboWriterTest', 'csvfile', 'sqlfile')
    Traceback (most recent call last):
    ...
    AssertionError: File not found: /Shared/johnsonhj/TrackOn/Experiments/20131119_TrackOn_ManualResults/HDNI_002/218087801/218087801_20120907_30/TissueClassify/t1_average_BRAINSABC.nii.gz

    >>> TODO:
    """
    values = {}
    labelKeys = LABELS.keys(); labelKeys.sort()
    try:
        b_T1_path, moving, t_T1_path, fixed = common(record, BENCHMARK, TEST, force)
    except AssertionError, err:
        if err.message.startswith('No benchmark'):
            for key in labelKeys:
                values[key] = flagForReview()
            values['t1_average'] = flagForReview()
            values['labels_tissue'] = flagForReview()
            values['t2_average'] = flagForReview()
            return writeReview(TEST, record, RESULTS, SQLOUT, values)
        elif err.message.startswith('No test'):
            print "Session was not processed in testing experiment"
            return 0

    (project, subject, session) = splitRecord(record)
    base = os.path.join(RESULTS, project, subject, session)
    tx_path = os.path.join(base, 'benchmarkToTest.mat')
    image_path = os.path.join(base, 'T1ToTest.nii.gz')
    resample_path = os.path.join(base, 'all_Labels_seg_resampled.nii.gz')
    cleaned_path = os.path.join(base, 'all_Labels_seg_roboRaterCleaned.nii.gz')
    if not force and os.path.exists(tx_path):
         benchmark2testTx = sitk.ReadTransform(tx_path)
    else:
        benchmark2testTx = transform(t_T1_path, b_T1_path, tx_path, image_path, force)
    if not force and os.path.exists(resample_path):
        resampledImg = sitk.ReadImage(resample_path)
    else:
        try:
            resampledImg = sitk.Resample(moving, fixed, benchmark2testTx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
        except:
            raise
        sitk.WriteImage(resampledImg, resample_path)
    compare = compare_labels(resampledImg, fixed)
    benchValues = []
    with open(CSVIN, 'rb') as csvin:
        reader = csv.DictReader(csvin, delimiter=',')
        for row in reader:
            benchValues.append(row)
    # Evaluate each label
    b_experiment = os.path.basename(BENCHMARK)
    for labelImageName in labelKeys:
        if compare[labelImageName]['dice'] >= DICE_THRESHOLD:
            values[labelImageName] = getQAResult(labelImageName, b_experiment, session)
        elif get_benchmark_quality(b_experiment, session, labelImageName, benchValues):
            # region in test is 'bad' compared to benchmark AND region in benchmark is 'good'
            if 'target' not in locals():
                target = sitk.Resample(resampledImg, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                combined = fixed
            print "Copying %s from benchmark to output" % labelImageName
            combined = modify_labels(target, combined, labelImageName)
            values[labelImageName] = 'SELECT qa_code FROM autoworkup_scm."QA_mapping" WHERE human = "roboWriter-acceptable"'
        else:
            values[labelImageName] = flagForReview()
    # Write final label image
    if 'combined' in locals():
        sitk.WriteImage(sitk.Cast(combined, sitk.sitkUInt8), cleaned_path)
    else:
        cleaned_path = resample_path
    assert os.path.exists(cleaned_path), "Registered test file must exist!"
    # Create the image review record
    values = getT1T2TissueLabels(b_experiment, session, values)
    return writeReview(TEST, record, RESULTS, SQLOUT, values)

def writeReview(TEST, record, RESULTS, SQLOUT, values):
    (project, subject, session) = splitRecord(record)
    results = setQAResult(os.path.basename(TEST), session, values)
    sqlDir = os.path.join(RESULTS, project, subject, session)
    if not os.path.isdir(sqlDir):
        os.makedirs(sqlDir)
    sqlFile = os.path.join(RESULTS, project, subject, session, SQLOUT)
    writeSQL(sqlFile, results)
    return 0


if __name__ == "__main__":
    import sys

    from docopt import docopt
    import ConfigParser

    args = docopt(__doc__)
    inputs = {}
    for key in args.keys():
        inputs[key.strip('-')] = args.pop(key)
    if inputs['test']:
        import doctest
        SQL_ACCESS = False
        doctest.testmod(verbose=inputs['verbose'])
    elif inputs['compare']:
        config = ConfigParser.SafeConfigParser()
        config.read(inputs['CONFIG'])
        retval = roboRater(config, **inputs)
        sys.exit(retval)
    elif inputs['write']:
        retval = roboWriter(**inputs)
        sys.exit(retval)
