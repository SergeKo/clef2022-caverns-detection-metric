import csv
import numpy as np


class CavernDetectiondEvaluator:
    """
    This class provides evaluation code which is used for scoring ImageCLEF 2022 Tuberculosis - Caverns Detection challenge.
    Please note, only the score calculation method is provided, prediction validation etc. is not included.
    """

    def __init__(self, ground_truth_path, all_ids_path, **kwargs):
        """
        `ground_truth_path` : Holds the path for the ground truth (GT) which is used to score the submissions.
                              The format for GT is same as train set metadata
        """
        self.ground_truth_path = ground_truth_path
        self.all_ids_path = all_ids_path
        self.gt = self.__load_gt__()
        self.all_IDs = self.__load_all_IDs__()

    def evaluate(self, submission_file_path):
        print("evaluating...")
        predictions = self.__load_predictions__(submission_file_path)
        return self.__compute_score__(predictions)

    def __load_gt__(self):
        print("loading ground truth...")

        gt = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            next(reader)  # skip header
            for row in reader:
                id = row[0] + '.nii.gz'
                coords = (int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]))
                if not id in gt.keys():
                    gt[id] = list()
                gt[id].append(coords)
        return gt

    def __load_all_IDs__(self):
        print("loading all IDs list...")

        all_ids = set()
        with open(self.all_ids_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                all_ids.add(row[0])
        return all_ids

    def __load_predictions__(self, submission_file_path):
        """
        Submit a plain text file with the following format:
        <Filename>,<Comma-separated coordinates (X1,Y1,Z1) of the first bounding box corner>,<Comma-separated coordinates (X2,Y2,Z2) of the second bounding box corner>
        e.g.:
            TST_001.nii.gz,10,10,200,20,20,235
            TST_002.nii.gz,100,100,200,110,110,220
            TST_002.nii.gz,150,150,210,176,176,210
            TST_005.nii.gz,123,123,123,200,200,200
            .....
        You need to respect the following constraints:
        - File should not have a header.
        - Filenames must be same as original test file names
        - Single case may have none, one or multiple bounding boxes. If none caverns were detected - file name should be absent in the submission file. If multiple caverns were detected - one line per each cavern should be present (see TST_002 in the example above).
        - All coordinates should be integer values inside bounds of the corresponding CT image.
        - Coordinates of corners should be ordered: X1 < X2, Y1 < Y2, Z1 < Z2
        """
        print("load predictions...")

        predictions = {}

        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

            for row in reader:
                id = row[0]
                coords = (int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]))  # tuple with 2 coordinates (6 values)
                if not id in predictions.keys():
                    predictions[id] = list()
                predictions[id].append(coords)

        # No predictions => Error
        if len(predictions) < 1:
            self.raise_exception("It seems you submitted an empty file.", 0)

        return predictions

    # helper function to calculate IoU
    def __iou__(self, box1, box2):
        x11, y11, z11, x12, y12, z12 = box1
        x21, y21, z21, x22, y22, z22 = box2

        vol1 = (x12 - x11) * (y12 - y11) * (z12 - z11)
        vol2 = (x22 - x21) * (y22 - y21) * (z22 - z21)

        xi1, yi1, zi1, xi2, yi2, zi2 = max([x11, x21]), max([y11, y21]), max([z11, z21]), min([x12, x22]), min(
            [y12, y22]), min([z12, z22])

        if xi2 <= xi1 or yi2 <= yi1 or zi2 <= zi1:
            return 0
        else:
            intersect = (xi2 - xi1) * (yi2 - yi1) * (zi2 - zi1)
            union = vol1 + vol2 - intersect
            return intersect / union

    def __map_iou__(self, boxes_true, boxes_pred, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
        """
        Based on: https://www.kaggle.com/code/chenyc15/mean-average-precision-metric/comments
        Mean average precision at different intersection over union (IoU) threshold for one CT case

        input:
            boxes_true: Mx6 numpy array of ground true bounding boxes of one image.
                        bbox format: (x1, y1, z1, x2, y2, z2)
            boxes_pred: Nx6 numpy array of predicted bounding boxes of one image.
                        bbox format: (x1, y1, z1, x2, y2, z2)
            thresholds: IoU thresholds to evaluate mean average precision on
        output:
            map: mean average precision of the image
        """
        assert boxes_true.shape[1] == 6 or boxes_pred.shape[1] == 6, "boxes should be 3D arrays with shape[1]=6"
        map_total = 0

        # loop over thresholds
        for t in thresholds:
            matched_bt = set()
            tp, fn = 0, 0
            for i, bt in enumerate(boxes_true):
                matched = False
                for j, bp in enumerate(boxes_pred):
                    miou = self.__iou__(bt, bp)
                    if miou >= t and not matched and j not in matched_bt:
                        matched = True
                        tp += 1  # bt is matched for the first time, count as TP
                        matched_bt.add(j)
                if not matched:
                    fn += 1  # bt has no match, count as FN

            fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
            m = tp / (tp + fn + fp)
            map_total += m

        return map_total / len(thresholds)

    def __compute_score__(self, predictions):
        """
        Compute and return the score
        """
        print("compute primary score...")
        MAP = 0
        for i, id in enumerate(self.gt.keys()):
            print(i, '\tprocessing boxes for case: ', id)
            if id in predictions:
                #if case GT has some cavern areas and prediction also has it, adding map_iou for given case
                id_boxes_pred = list()
                for p in predictions[id]:
                    id_boxes_pred.append(p)
                print('ground truth: ', self.gt[id])
                print('predicted:', id_boxes_pred)
                MAP += self.__map_iou__(np.array(self.gt[id]), np.array(id_boxes_pred))

        for id in self.all_IDs:
            if not id in predictions and not id in self.gt:
                # if case GT has NO cavern areas and prediction also has NO areas for given case
                # (TRUE NEGATIVE CASE) - adding max score for current image
                MAP += 1

        return MAP / len(self.all_IDs)


# TEST THIS EVALUATOR
if __name__ == "__main__":
    all_ids_path = "sample/all_ids.csv"  # file with all CT IDs presented in challenge train data
    ground_truth_path = "sample/gt_caverns_detection_train_bboxes.csv" # bounding boxes from challenge train data
    submission_file_path = "sample/prediction.csv"  # same bounding boxes from challenge train data
                                                    # converted to submission format

    evaluator = CavernDetectiondEvaluator(ground_truth_path, all_ids_path)
    score = evaluator.evaluate(submission_file_path)
    print('\nScore: ', score)
