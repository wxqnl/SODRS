
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling import ops
from functools import partial

__all__ = ['FCOSLoss', 'FCOSLossMILC', 'FCOSLoss_SODRS']


def flatten_tensor(inputs, channel_first=False):
    """
    Flatten a Tensor
    Args:
        inputs (Tensor): 4-D Tensor with shape [N, C, H, W] or [N, H, W, C]
        channel_first (bool): If true the dimension order of Tensor is 
            [N, C, H, W], otherwise is [N, H, W, C]
    Return:
        output_channel_last (Tensor): The flattened Tensor in channel_last style
    """
    if channel_first:
        input_channel_last = paddle.transpose(inputs, perm=[0, 2, 3, 1])
    else:
        input_channel_last = inputs
    output_channel_last = paddle.flatten(
        input_channel_last, start_axis=0, stop_axis=2)
    return output_channel_last


@register
class FCOSLoss(nn.Layer):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type (str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights (float): weight for location loss
        quality (str): quality branch, centerness/iou
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0,
                 quality='centerness'):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights
        self.quality = quality

    def _iou_loss(self,
                  pred,
                  targets,
                  positive_mask,
                  weights=None,
                  return_iou=False):
        """
        Calculate the loss for location prediction
        Args:
            pred (Tensor): bounding boxes prediction
            targets (Tensor): targets for positive samples
            positive_mask (Tensor): mask of positive samples
            weights (Tensor): weights for each positive samples
        Return:
            loss (Tensor): location loss
        """
        plw = pred[:, 0] * positive_mask
        pth = pred[:, 1] * positive_mask
        prw = pred[:, 2] * positive_mask
        pbh = pred[:, 3] * positive_mask

        tlw = targets[:, 0] * positive_mask
        tth = targets[:, 1] * positive_mask
        trw = targets[:, 2] * positive_mask
        tbh = targets[:, 3] * positive_mask
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        clw = paddle.maximum(plw, tlw)
        crw = paddle.maximum(prw, trw)
        cth = paddle.maximum(pth, tth)
        cbh = paddle.maximum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
                area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask

        if return_iou:
            return ious

        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - paddle.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels,
                tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Tensor, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Tensor, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Tensor, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Tensor, which is category
                targets for each anchor point
            tag_bboxes (list): list of Tensor, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Tensor, which is centerness
                targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                flatten_tensor(cls_logits[lvl], True))
            bboxes_reg_flatten_list.append(
                flatten_tensor(bboxes_reg[lvl], True))
            centerness_flatten_list.append(
                flatten_tensor(centerness[lvl], True))

            tag_labels_flatten_list.append(

                flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(
                flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                flatten_tensor(tag_center[lvl], False))

        cls_logits_flatten = paddle.concat(cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = paddle.concat(bboxes_reg_flatten_list, axis=0)
        centerness_flatten = paddle.concat(centerness_flatten_list, axis=0)

        tag_labels_flatten = paddle.concat(tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = paddle.concat(tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = paddle.concat(tag_center_flatten_list, axis=0)
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True

        mask_positive_bool = tag_labels_flatten > 0
        mask_positive_bool.stop_gradient = True
        mask_positive_float = paddle.cast(mask_positive_bool, dtype="float32")
        mask_positive_float.stop_gradient = True

        num_positive_fp32 = paddle.sum(mask_positive_float)
        num_positive_fp32.stop_gradient = True
        num_positive_int32 = paddle.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_int32.stop_gradient = True

        normalize_sum = paddle.sum(tag_center_flatten * mask_positive_float)
        normalize_sum.stop_gradient = True

        # 1. cls_logits: sigmoid_focal_loss
        # expand onehot labels
        num_classes = cls_logits_flatten.shape[-1]
        tag_labels_flatten = paddle.squeeze(tag_labels_flatten, axis=-1)
        tag_labels_flatten_bin = F.one_hot(
            tag_labels_flatten, num_classes=1 + num_classes)
        tag_labels_flatten_bin = tag_labels_flatten_bin[:, 1:]
        # sigmoid_focal_loss
        cls_loss = F.sigmoid_focal_loss(
            cls_logits_flatten, tag_labels_flatten_bin) / num_positive_fp32

        if self.quality == 'centerness':
            # 2. bboxes_reg: giou_loss
            mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
            tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
            reg_loss = self._iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=tag_center_flatten)
            reg_loss = reg_loss * mask_positive_float / normalize_sum

            # 3. centerness: sigmoid_cross_entropy_with_logits_loss
            centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
            quality_loss = ops.sigmoid_cross_entropy_with_logits(
                centerness_flatten, tag_center_flatten)
            quality_loss = quality_loss * mask_positive_float / num_positive_fp32

        elif self.quality == 'iou':
            # 2. bboxes_reg: giou_loss
            mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
            tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
            reg_loss = self._iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=None)
            reg_loss = reg_loss * mask_positive_float / num_positive_fp32
            # num_positive_fp32 is num_foreground

            # 3. centerness: sigmoid_cross_entropy_with_logits_loss
            centerness_flatten = paddle.squeeze(centerness_flatten, axis=-1)
            gt_ious = self._iou_loss(
                bboxes_reg_flatten,
                tag_bboxes_flatten,
                mask_positive_float,
                weights=None,
                return_iou=True)
            quality_loss = ops.sigmoid_cross_entropy_with_logits(
                centerness_flatten, gt_ious)
            quality_loss = quality_loss * mask_positive_float / num_positive_fp32
        else:
            raise Exception(f'Unknown quality type: {self.quality}')

        loss_all = {
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss),
            "loss_quality": paddle.sum(quality_loss),
        }
        return loss_all


@register
class FCOSLossMILC(FCOSLoss):
    """
    FCOSLossMILC for ARSL in semi-det(ssod)
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type (str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights (float): weight for location loss
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="giou",
                 reg_weights=1.0,
                 num_classes=15):
        super(FCOSLossMILC, self).__init__()
        self.loss_alpha = loss_alpha
        self.num_classes = num_classes
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def crf_post_process(self, bboxes, labels, emissions):
        # Perform CRF post-processing on predicted bounding boxes and labels
        new_labels = crf_post_processs(bboxes, labels, self.num_classes, emissions)
        return new_labels

    def iou_loss(self, pred, targets, weights=None, avg_factor=None):
        """
        Calculate the loss for location prediction
        Args:
            pred (Tensor): bounding boxes prediction
            targets (Tensor): targets for positive samples
            weights (Tensor): weights for each positive samples
        Return:
            loss (Tensor): location loss
        """
        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        clw = paddle.maximum(plw, tlw)
        crw = paddle.maximum(prw, trw)
        cth = paddle.maximum(pth, tth)
        cbh = paddle.maximum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
                area_predict + area_target - area_inter + 1.0)
        ious = ious

        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - paddle.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # temp function: calcualate iou between bbox and target
    def _bbox_overlap_align(self, pred, targets):
        assert pred.shape[0] == targets.shape[0], \
            'the pred should be aligned with target.'

        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
                area_predict + area_target - area_inter + 1.0)

        return ious

    def iou_based_soft_label_loss(self, pred, target, alpha=0.75, gamma=2.0, iou_weighted=False, implicit_iou=None,
                                  avg_factor=None):
        """
        Calculate the soft label loss with enhanced focal weight adjustment for better handling of遥感 data.
        """
        assert pred.shape == target.shape
        pred = F.sigmoid(pred)  # Ensure predictions are in the range [0, 1]
        target = target.cast(pred.dtype)  # Ensure data type consistency

        if implicit_iou is not None:
            pred = pred * implicit_iou  # Adjust predictions by implicit IoU

        # Calculate the element-wise product for positive and negative parts
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1.0 - pt).pow(gamma)  # Calculate focal weight based on prediction accuracy

        # Calculate basic binary cross entropy loss
        loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight

        if iou_weighted:
            # Adjust focal weight if IoU weighted loss is required
            adjusted_weight = (pred - target).abs().pow(gamma) * target + \
                              alpha * (pred - target).abs().pow(gamma) * (1 - target)
            loss = loss * adjusted_weight  # Apply adjusted weight to loss
        # else:
        # Apply the adjusted focal weight if iou_weighted is False
        #   adjusted_weight = (pred - target).abs().pow(gamma) + \
        #      alpha * (pred - target).abs().pow(gamma) * (1 - target)
        # loss = loss * adjusted_weight  # Apply adjusted weight to loss

        if avg_factor is not None:
            loss = loss / avg_factor  # Normalize loss by average factor
        # else:
        #   loss = loss.mean()  # Default to mean loss if avg_factor is not provided

        # Scale the loss to ensure it is effective for遥感 data
        # loss = loss * self.loss_alpha

        return loss

    def calculate_iou(self, pred_boxes, true_boxes):
        """
        Calculate Intersection over Union (IoU) for predicted and true boxes.
        """
        # Get the coordinates of bounding boxes
        x1_pred, y1_pred, x2_pred, y2_pred = paddle.unbind(pred_boxes, axis=1)
        x1_true, y1_true, x2_true, y2_true = paddle.unbind(true_boxes, axis=1)

        # Calculate the intersection rectangles
        max_x1 = paddle.maximum(x1_pred, x1_true)
        max_y1 = paddle.maximum(y1_pred, y1_true)
        min_x2 = paddle.minimum(x2_pred, x2_true)
        min_y2 = paddle.minimum(y2_pred, y2_true)

        # Calculate intersection area
        intersection = paddle.maximum(0.0, min_x2 - max_x1) * paddle.maximum(0.0, min_y2 - max_y1)

        # Calculate union area
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        true_area = (x2_true - x1_true) * (y2_true - y1_true)
        union = pred_area + true_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-7)
        return iou

    def calculate_center_distance(self, pred_boxes, true_boxes):
        """
        Calculate the Euclidean distance between the centers of predicted and true boxes.
        """
        pred_centers = (pred_boxes[..., 2:] + pred_boxes[..., :2]) / 2
        true_centers = (true_boxes[..., 2:] + true_boxes[..., :2]) / 2
        center_distance = paddle.norm(pred_centers[:, None, :] - true_centers[None, :, :], p=2, axis=-1)
        return center_distance

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels,
                tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Tensor, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Tensor, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Tensor, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Tensor, which is category
                targets for each anchor point
            tag_bboxes (list): list of Tensor, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Tensor, which is centerness
                targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                flatten_tensor(cls_logits[lvl], True))
            bboxes_reg_flatten_list.append(
                flatten_tensor(bboxes_reg[lvl], True))
            centerness_flatten_list.append(
                flatten_tensor(centerness[lvl], True))

            tag_labels_flatten_list.append(
                flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(
                flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                flatten_tensor(tag_center[lvl], False))

        cls_logits_flatten = paddle.concat(cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = paddle.concat(bboxes_reg_flatten_list, axis=0)
        centerness_flatten = paddle.concat(centerness_flatten_list, axis=0)

        tag_labels_flatten = paddle.concat(tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = paddle.concat(tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = paddle.concat(tag_center_flatten_list, axis=0)
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True

        # find positive index
        mask_positive_bool = tag_labels_flatten > 0
        mask_positive_bool.stop_gradient = True
        mask_positive_float = paddle.cast(mask_positive_bool, dtype="float32")
        mask_positive_float.stop_gradient = True

        num_positive_fp32 = paddle.sum(mask_positive_float)
        num_positive_fp32.stop_gradient = True
        num_positive_int32 = paddle.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_int32.stop_gradient = True

        # centerness target is used as reg weight
        normalize_sum = paddle.sum(tag_center_flatten * mask_positive_float)
        normalize_sum.stop_gradient = True

        # 1. IoU-Based soft label loss
        # calculate iou
        with paddle.no_grad():
            pos_ind = paddle.nonzero(
                tag_labels_flatten.reshape([-1]) > 0).reshape([-1])
            pos_pred = bboxes_reg_flatten[pos_ind]
            pos_target = tag_bboxes_flatten[pos_ind]
            bbox_iou = self._bbox_overlap_align(pos_pred, pos_target)
        # pos labels
        pos_labels = tag_labels_flatten[pos_ind].squeeze(1)
        cls_target = paddle.zeros(cls_logits_flatten.shape)
        cls_target[pos_ind, pos_labels - 1] = bbox_iou
        cls_loss = self.iou_based_soft_label_loss(
            cls_logits_flatten,
            cls_target,
            implicit_iou=F.sigmoid(centerness_flatten),
            avg_factor=num_positive_fp32)

        # 2. bboxes_reg: giou_loss
        mask_positive_float = paddle.squeeze(mask_positive_float, axis=-1)
        tag_center_flatten = paddle.squeeze(tag_center_flatten, axis=-1)
        reg_loss = self._iou_loss(
            bboxes_reg_flatten,
            tag_bboxes_flatten,
            mask_positive_float,
            weights=tag_center_flatten)
        reg_loss = reg_loss * mask_positive_float / normalize_sum

        # 3. iou loss
        pos_iou_pred = paddle.squeeze(centerness_flatten, axis=-1)[pos_ind]
        loss_iou = ops.sigmoid_cross_entropy_with_logits(pos_iou_pred, bbox_iou)
        loss_iou = loss_iou / num_positive_fp32 * 0.5

        loss_all = {
            "loss_cls": paddle.sum(cls_loss),
            "loss_box": paddle.sum(reg_loss),
            'loss_iou': paddle.sum(loss_iou),
        }

        return loss_all


# Concat multi-level feature maps by image
def levels_to_images(mlvl_tensor):
    batch_size = mlvl_tensor[0].shape[0]
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].shape[1]
    for t in mlvl_tensor:
        t = t.transpose([0, 2, 3, 1])
        t = t.reshape([batch_size, -1, channels])
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [paddle.concat(item, axis=0) for item in batch_list]


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def fill_diagonal(matrix, value):
    """
    在给定的方阵矩阵中填充对角线上的元素为指定的值。

    参数:
        matrix (paddle.Tensor): 一个方阵张量。
        value (float): 要填充对角线的值。

    返回:
        paddle.Tensor: 对角线元素被填充的矩阵。
    """
    # 获取矩阵的大小
    n = matrix.shape[0]
    # 直接在原矩阵上修改对角线元素的值
    for i in range(n):
        matrix[i, i] = value
    return matrix


def mean_field_crf_decoding(emissions, transition_matrix, label_length=None, num_iterations=5):
    """
    Perform CRF decoding using the mean field algorithm.
    :param emissions: The emission probabilities, shape is [batch_size, seq_len, num_tags]
    :param transition_matrix: The transition matrix, shape is [num_tags, num_tags]
    :param label_length: The actual length of each sequence, shape is [batch_size,]
    :param num_iterations: The number of iterations to run the mean field algorithm
    :return: The most likely tag sequence for each example, shape is [batch_size, seq_len]
    """
    batch_size, seq_len, num_tags = emissions.shape

    # Initialize the tag probabilities
    tag_probabilities = paddle.ones_like(emissions)  # Initialize as a copy of emissions
    tag_probabilities -= 1e30  # Start with a very low probability
    tag_probabilities += emissions  # Set the initial probability based on emissions

    for iteration in range(num_iterations):
        prev_prob = tag_probabilities.clone()

        for b in range(batch_size):
            for t in range(seq_len):
                if label_length is not None and t >= label_length[b]:
                    continue  # Skip padding
                for tag in range(num_tags):
                    emissions_contribution = emissions[b, t, tag]
                    transition_contribution = paddle.sum(tag_probabilities[b, t, :] * transition_matrix[tag])
                    tag_probabilities[b, t, tag] = emissions_contribution + transition_contribution

        # Normalize probabilities
        tag_probabilities = tag_probabilities - paddle.log(
            paddle.sum(paddle.exp(tag_probabilities), axis=-1, keepdim=True))

    mask = paddle.arange(seq_len).expand((batch_size, -1)) < (
        label_length.unsqueeze(1) if label_length is not None else paddle.full((batch_size, seq_len), seq_len,
                                                                               dtype=paddle.int64))
    mask = mask.astype('bool')

    _, tag_seq = paddle.max(tag_probabilities.unsqueeze(1).masked_fill(~mask.unsqueeze(-1), -1e30), axis=-1)
    tag_seq = paddle.squeeze(tag_seq, axis=1)

    return tag_seq


def local_decoding(emissions, transition_matrix, label_length=None):
    """
    Perform CRF decoding using a simplified local decoding approach.
    :param emissions: The emission probabilities, shape is [batch_size, seq_len, num_tags]
    :param transition_matrix: The transition matrix, shape is [num_tags, num_tags]
    :param label_length: The actual length of each sequence, shape is [batch_size,]
    :return: The most likely tag sequence for each example, shape is [batch_size, seq_len]
    """
    with paddle.amp.auto_cast():
        batch_size, seq_len, num_tags = emissions.shape
        # print(f"Shape of emissions: {emissions.shape}")
        # print(f"Shape of transition_matrix: {transition_matrix.shape}")

        # Initialize the tag sequence with the first tag of highest emission probability
        tag_seq = paddle.argmax(emissions[:, 0, :], axis=-1).unsqueeze(1)  # [batch_size, 1]

        # print(f"Initial tag_seq shape: {tag_seq.shape}")

        # Loop through each time step in the sequence
        for t in range(1, seq_len):
            emissions_t = emissions[:, t, :]
            previous_tags = tag_seq[:, -1]  # Shape [batch_size]

            # print(f"emissions_t shape at time {t}: {emissions_t.shape}")
            # print(f"previous_tags shape at time {t}: {previous_tags.shape}")

            # Calculate the score for each tag based on the transition and emission scores
            scores = paddle.zeros((batch_size, num_tags))
            for tag in range(num_tags):
                transition_score = paddle.sum(previous_tags.unsqueeze(-1) * transition_matrix[:, tag].unsqueeze(0),
                                              axis=-1)
                scores[:, tag] = emissions_t[:, tag] + transition_score

            # Select the tag with the highest score
            current_tag = paddle.argmax(scores, axis=-1)
            tag_seq = paddle.concat([tag_seq, current_tag.unsqueeze(1)], axis=1)

        # print(f"current_tag shape at time {t}: {current_tag.shape}")
        # print(f"tag_seq shape at time {t}: {tag_seq.shape}")

        if label_length is not None:
            max_label_length = paddle.max(label_length).item()  # Get the maximum length as a Python integer
            tag_seq = tag_seq[:, :max_label_length]  # Trim the sequences to their true length

    return tag_seq.squeeze(-1)


def crf_decoding(emissions, transition_matrix, label_length=None):
    """
    实现 CRF 解码
    :param emissions: 形状为 (batch_size, seq_len, num_tags) 的发射概率
    :param transition_matrix: 形状为 (batch_size, num_tags, num_tags) 的转移矩阵
    :param label_length: 每个序列的实际长度，形状为 (batch_size,)
    :return: 解码后的标签序列，形状为 (batch_size, seq_len)
    """
    batch_size, seq_len, num_tags = emissions.shape

    # 初始化前向变量
    viterbi = paddle.zeros((batch_size, seq_len, num_tags))
    backpointers = paddle.zeros((batch_size, seq_len, num_tags), dtype=paddle.int64)

    # 初始状态
    viterbi[:, 0, :] = emissions[:, 0, :]
    with paddle.no_grad(), auto_cast():
        for step in range(1, seq_len):
            # 广播操作
            viterbi_step = viterbi[:, step - 1, :].unsqueeze(-1)  # (batch_size, num_tags, 1)
            transition_step = transition_matrix  # (batch_size, num_tags, num_tags)
            scores = viterbi_step + transition_step  # (batch_size, num_tags, num_tags)

            # 获取最大值和对应的索引
            max_scores, best_tags = paddle.max(scores, axis=1), paddle.argmax(scores, axis=1)  # (batch_size, num_tags)

            viterbi[:, step, :] = max_scores + emissions[:, step, :]
            backpointers[:, step, :] = best_tags

        # 回溯解码
        best_paths = []
        for i in range(batch_size):
            if label_length is not None:
                seq_len_i = label_length[i].item()
            else:
                seq_len_i = seq_len

            best_path = [paddle.argmax(viterbi[i, seq_len_i - 1, :]).item()]
            for step in range(seq_len_i - 1, 0, -1):
                best_tag = best_path[-1]
                best_path.append(backpointers[i, step, best_tag].item())
            best_path.reverse()
            best_paths.append(best_path)

    return paddle.to_tensor(best_paths, dtype=paddle.int64)


def crf_post_processs(bboxes, labels, num_classes, emissions):
    with paddle.no_grad(), paddle.amp.auto_cast():
        batch_size = emissions.shape[0]

        # Initialize the transition matrix for CRF
        transition = paddle.full((num_classes, num_classes), 0.1, dtype='float16')
        transition = fill_diagonal(transition, 0.5)
        transition = paddle.tile(transition.unsqueeze(0), [batch_size, 1, 1])

        # Calculate pairwise potentials based on bboxes
        pairwise_potentials = calculate_pairwise_potentials(bboxes)
        labels = labels.astype('int64')  # Ensure labels are of type int64
        labels_one_hot = paddle.nn.functional.one_hot(labels, num_classes=num_classes)
        labels_one_hot_sum = paddle.sum(labels_one_hot, axis=-1)
        emissions = emissions + labels_one_hot_sum.astype('float16')

        # Reshape transition matrix to match emissions shape
        transition_reshaped = paddle.squeeze(transition, axis=1)  # Shape becomes [batch_size, num_classes]

        # Expand pairwise potentials to match emissions shape
        pairwise_potentials_expanded = []
        target_size = num_classes
        for i in range(batch_size):
            pairwise_potentials_i = pairwise_potentials[i].unsqueeze(0)
            if pairwise_potentials_i.shape[1] != emissions.shape[1]:
                repeat_times = int(target_size / pairwise_potentials_i.shape[1])
                if target_size % pairwise_potentials_i.shape[1] != 0:
                    repeat_times += 1  # 如果除不尽，需要额外重复一次
                pairwise_potentials_i = paddle.tile(pairwise_potentials_i, [1, repeat_times])
                # 确保不会超过目标大小
                pairwise_potentials_i = pairwise_potentials_i[:, :target_size]
            pairwise_potentials_expanded.append(pairwise_potentials_i)
        pairwise_potentials_expanded = paddle.concat(pairwise_potentials_expanded, axis=0)

        # Combine potentials
        emissions_expanded = emissions.unsqueeze(-1)
        # print(emissions.shape)
        # print(transition_reshaped.shape)
        # print(pairwise_potentials_expanded.shape)

        combined_potentials = emissions_expanded + transition_reshaped + pairwise_potentials_expanded.unsqueeze(-1)
        # combined_potentials = emissions + transition_reshaped + pairwise_potentials_expanded

        # CRF Decoding
        label_length = paddle.full([batch_size], emissions.shape[1], dtype='int64')
        # print(2)
        decoded_labels = local_decoding(combined_potentials, transition, label_length)
        # print(1)
    # print(decoded_labels)
    return decoded_labels


def calculate_pairwise_potentials(bboxes):
    """
    Calculate pairwise potentials based on the spatial relationships between bounding boxes
    Args:
        bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_bboxes, 4]
    Return:
        pairwise_potentials (Tensor): Pairwise potentials with shape [batch_size, num_bboxes, num_bboxes]
    """

    pairwise_potentials = paddle.zeros_like(bboxes)
    return pairwise_potentials


@register
class FCOSLoss_SODRS(FCOSLossMILC):
    """
    FCOSLoss of Consistency Regularization
    """

    def __init__(self,
                 iou_loss_type="giou",
                 cls_weight=2.0,
                 reg_weight=2.0,
                 iou_weight=0.5,
                 hard_neg_mining_flag=True,
                 num_classes=15):
        super(FCOSLoss_SODRS, self).__init__()
        self.num_classes = num_classes
        self.iou_loss_type = iou_loss_type
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.hard_neg_mining_flag = hard_neg_mining_flag

    def crf_post_process(self, bboxes, labels, emissions):
        """
        Perform CRF post-processing on predicted bounding boxes and labels
        Args:
            bboxes (Tensor): Predicted bounding boxes
            labels (Tensor): Predicted labels
            num_classes (int): Number of classes
        Return:
            new_labels (Tensor): Labels after CRF post-processing
        """
        # Get the emission scores from the model predictions
        # This could be logits from the classification head, which we can softmax to get probabilities

        # Perform the CRF post-processing
        new_labels = crf_post_processs(bboxes, labels, self.num_classes, emissions)

        # Optionally, you can also update the bboxes here based on the new labels
        # For example, you could filter out bboxes that have very low max label probability
        prob_threshold = 0.5
        new_labels_float = paddle.cast(new_labels, dtype='float32')
        probabilities = F.softmax(new_labels_float, axis=1)
        max_prob = paddle.max(probabilities, axis=-1, keepdim=True)
        keep_mask = max_prob > prob_threshold
        # new_bboxes = bboxes[keep_mask]
        # print("bboxes shape:", bboxes.shape)
        # print("keep_mask shape:", keep_mask.shape)

        if keep_mask.shape != (bboxes.shape[0],):
            # 调整 keep_mask 以匹配 bboxes 的第一维
            keep_mask = keep_mask.flatten()

        # 使用 keep_mask 来过滤 bboxes
        new_bboxes = paddle.masked_select(bboxes, keep_mask.unsqueeze(-1).expand_as(bboxes))
        return new_bboxes, new_labels

    def iou_loss(self, pred, targets, weights=None, avg_factor=None):
        """
            Calculate the loss for location prediction
            Args:
                pred (Tensor): bounding boxes prediction
                targets (Tensor): targets for positive samples
                weights (Tensor): weights for each positive samples
            Return:
                loss (Tensor): location loss
            """
        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        clw = paddle.maximum(plw, tlw)
        crw = paddle.maximum(prw, trw)
        cth = paddle.maximum(pth, tth)
        cbh = paddle.maximum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
                area_predict + area_target - area_inter + 1.0)
        ious = ious

        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - paddle.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # calcualate iou between bbox and target
    def bbox_overlap_align(self, pred, targets):
        assert pred.shape[0] == targets.shape[0], \
            'the pred should be aligned with target.'

        plw = pred[:, 0]
        pth = pred[:, 1]
        prw = pred[:, 2]
        pbh = pred[:, 3]

        tlw = targets[:, 0]
        tth = targets[:, 1]
        trw = targets[:, 2]
        tbh = targets[:, 3]

        ilw = paddle.minimum(plw, tlw)
        irw = paddle.minimum(prw, trw)
        ith = paddle.minimum(pth, tth)
        ibh = paddle.minimum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
        ious = (area_inter + 1.0) / (
                area_predict + area_target - area_inter + 1.0)
        return ious

    # cls loss: iou-based soft lable with joint iou
    def quality_focal_loss(self,
                           stu_cls,
                           targets,
                           quality=None,
                           weights=None,
                           alpha=0.75,
                           gamma=2.0,
                           avg_factor='sum'):
        stu_cls = F.sigmoid(stu_cls)
        if quality is not None:
            stu_cls = stu_cls * F.sigmoid(quality)
        num_classes = 80
        print(targets)
        # 将 targets 转换为 one-hot 编码
        targets_rounded = paddle.round(targets)
        targets = paddle.cast(targets_rounded, dtype='int64')
        print(targets)
        targets = paddle.nn.functional.one_hot(targets, num_classes=num_classes)
        targets = targets.astype('float32')
        print(targets)
        focal_weight = (stu_cls - targets).abs().pow(gamma) * (targets > 0.0).cast('float32') + \
                       alpha * (stu_cls - targets).abs().pow(gamma) * \
                       (targets <= 0.0).cast('float32')

        loss = F.binary_cross_entropy(
            stu_cls, targets, reduction='none') * focal_weight

        if weights is not None:
            loss = loss * weights.reshape([-1, 1])
        loss = paddle.sum(loss)
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss

    # generate points according to feature maps
    def compute_locations_by_level(self, fpn_stride, h, w):
        """
        Compute locations of anchor points of each FPN layer
        Return:
            Anchor points locations of current FPN feature map
        """
        shift_x = paddle.arange(0, w * fpn_stride, fpn_stride)
        shift_y = paddle.arange(0, h * fpn_stride, fpn_stride)
        shift_x = paddle.unsqueeze(shift_x, axis=0)
        shift_y = paddle.unsqueeze(shift_y, axis=1)
        shift_x = paddle.expand(shift_x, shape=[h, w])
        shift_y = paddle.expand(shift_y, shape=[h, w])
        shift_x = paddle.reshape(shift_x, shape=[-1])
        shift_y = paddle.reshape(shift_y, shape=[-1])
        location = paddle.stack(
            [shift_x, shift_y], axis=-1) + float(fpn_stride) / 2
        return location

    # decode bbox from ltrb to x1y1x2y2
    def decode_bbox(self, ltrb, points):
        assert ltrb.shape[0] == points.shape[0], \
            "When decoding bbox in one image, the num of loc should be same with points."
        bbox_decoding = paddle.stack(
            [
                points[:, 0] - ltrb[:, 0], points[:, 1] - ltrb[:, 1],
                points[:, 0] + ltrb[:, 2], points[:, 1] + ltrb[:, 3]
            ],
            axis=1)
        return bbox_decoding

    # encode bbox from x1y1x2y2 to ltrb
    def encode_bbox(self, bbox, points):
        assert bbox.shape[0] == points.shape[0], \
            "When encoding bbox in one image, the num of bbox should be same with points."
        bbox_encoding = paddle.stack(
            [
                points[:, 0] - bbox[:, 0], points[:, 1] - bbox[:, 1],
                bbox[:, 2] - points[:, 0], bbox[:, 3] - points[:, 1]
            ],
            axis=1)
        return bbox_encoding

    def calcualate_iou(self, gt_bbox, predict_bbox):
        # bbox area
        gt_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * \
                  (gt_bbox[:, 3] - gt_bbox[:, 1])
        predict_area = (predict_bbox[:, 2] - predict_bbox[:, 0]) * \
                       (predict_bbox[:, 3] - predict_bbox[:, 1])
        # overlop area
        lt = paddle.fmax(gt_bbox[:, None, :2], predict_bbox[None, :, :2])
        rb = paddle.fmin(gt_bbox[:, None, 2:], predict_bbox[None, :, 2:])
        wh = paddle.clip(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # iou
        iou = overlap / (gt_area[:, None] + predict_area[None, :] - overlap)
        return iou

    # select potential positives from hard negatives
    import paddle

    def hard_neg_mining(self, cls_score, loc_ltrb, quality, pos_ind, hard_neg_ind, loc_mask, loc_targets,
                        num_iterations=5, confidence_threshold=0.5):
        # 获取正样本和硬负样本的特征
        pos_features = paddle.concat((cls_score[pos_ind], loc_ltrb[pos_ind], quality[pos_ind]), axis=-1)
        neg_features = paddle.concat((cls_score[hard_neg_ind], loc_ltrb[hard_neg_ind], quality[hard_neg_ind]), axis=-1)
        # 获取正样本和硬负样本的特征

        # 构建图的邻接矩阵
        pos_dists = paddle.norm(pos_features[:, None] - pos_features[None, :], axis=2)
        neg_dists = paddle.norm(neg_features[:, None] - neg_features[None, :], axis=2)

        # 使用高斯核计算相似度
        pos_similarities = paddle.exp(-pos_dists)
        neg_similarities = paddle.exp(-neg_dists)

        # 归一化相似度矩阵
        pos_similarities /= paddle.sum(pos_similarities, axis=1, keepdim=True)
        neg_similarities /= paddle.sum(neg_similarities, axis=1, keepdim=True)

        # 初始化传播标签的矩阵
        pos_label_matrix = paddle.zeros_like(pos_similarities)
        neg_label_matrix = paddle.zeros_like(neg_similarities)

        # 创建对角线标签矩阵
        for i in range(pos_features.shape[0]):
            pos_label_matrix[i, i] = 1

        # 标签传播迭代
        for _ in range(num_iterations):
            pos_label_matrix = paddle.matmul(pos_similarities, pos_label_matrix)
            neg_label_matrix = paddle.matmul(neg_similarities, neg_label_matrix)

        # 选择负样本
        pos_scores = paddle.sum(pos_label_matrix, axis=1)
        neg_scores = paddle.sum(neg_label_matrix, axis=1)

        # 选择分数超过置信度阈值的负样本
        selected_hard_neg_ind = []
        for i, score in enumerate(neg_scores):
            if score > confidence_threshold and pos_scores[i] > confidence_threshold:
                selected_hard_neg_ind.append(hard_neg_ind[i])

        if not selected_hard_neg_ind:
            return loc_mask, loc_targets
        # 更新loc_mask和loc_targets

        loc_mask[selected_hard_neg_ind] = 1.
        loc_targets[selected_hard_neg_ind] = loc_ltrb[selected_hard_neg_ind]

        return loc_mask, loc_targets

    def is_inside(self, pos_bbox, potential_pos_bbox):
        lt = paddle.maximum(pos_bbox[:, None, :2], potential_pos_bbox[None, :, :2])
        rb = paddle.minimum(pos_bbox[:, None, 2:], potential_pos_bbox[None, :, 2:])
        wh = paddle.clip(rb - lt, min=0)
        return wh[:, :, 0] * wh[:, :, 1] > 0

    # get training targets
    def get_targets_per_img(self, tea_cls, tea_loc, tea_iou, stu_cls, stu_loc,
                            stu_iou):

        ### sample selection
        # prepare datas
        tea_cls_scores = F.sigmoid(tea_cls) * F.sigmoid(tea_iou)
        class_ind = paddle.argmax(tea_cls_scores, axis=-1)
        max_vals = paddle.max(tea_cls_scores, axis=-1)
        cls_mask = paddle.zeros_like(
            max_vals
        )  # set cls valid mask: pos is 1, hard_negative and negative are 0.
        num_pos, num_hard_neg = 0, 0

        # mean-std selection
        # using nonzero to turn index from bool to int, because the index will be used to compose two-dim index in following.
        # using squeeze rather than reshape to avoid errors when no score is larger than thresh.
        candidate_ind = paddle.nonzero(max_vals >= 0.1).squeeze(axis=-1)
        num_candidate = candidate_ind.shape[0]
        if num_candidate > 0:
            # pos thresh = mean + std to select pos samples
            candidate_score = max_vals[candidate_ind]
            candidate_score_mean = candidate_score.mean()
            candidate_score_std = candidate_score.std()
            pos_thresh = (candidate_score_mean + candidate_score_std).clip(
                max=0.4)
            # select pos
            pos_ind = paddle.nonzero(max_vals >= pos_thresh).squeeze(axis=-1)
            num_pos = pos_ind.shape[0]
            # select hard negatives as potential pos
            hard_neg_ind = (max_vals >= 0.1) & (max_vals < pos_thresh)
            hard_neg_ind = paddle.nonzero(hard_neg_ind).squeeze(axis=-1)
            num_hard_neg = hard_neg_ind.shape[0]
        # if not positive, directly select top-10 as pos.
        if (num_pos == 0):
            num_pos = 10
            _, pos_ind = paddle.topk(max_vals, k=num_pos)
        cls_mask[pos_ind] = 1.

        ### Consistency Regularization Training targets
        # cls targets
        pos_class_ind = class_ind[pos_ind]
        cls_targets = paddle.zeros_like(tea_cls)
        cls_targets[pos_ind, pos_class_ind] = tea_cls_scores[pos_ind,
        pos_class_ind]
        # hard negative cls target
        if num_hard_neg != 0:
            cls_targets[hard_neg_ind] = tea_cls_scores[hard_neg_ind]
        # loc targets
        loc_targets = paddle.zeros_like(tea_loc)
        loc_targets[pos_ind] = tea_loc[pos_ind]
        # iou targets
        iou_targets = paddle.zeros(
            shape=[tea_iou.shape[0]], dtype=tea_iou.dtype)
        iou_targets[pos_ind] = F.sigmoid(
            paddle.squeeze(
                tea_iou, axis=-1)[pos_ind])

        loc_mask = cls_mask.clone()
        # select potential positive from hard negatives for loc_task training
        if (num_hard_neg > 0) and self.hard_neg_mining_flag:
            results = self.hard_neg_mining(tea_cls, tea_loc, tea_iou, pos_ind,
                                           hard_neg_ind, loc_mask, loc_targets)
            if results is not None:
                loc_mask, loc_targets = results
                loc_pos_ind = paddle.nonzero(loc_mask > 0.).squeeze(axis=-1)
                iou_targets[loc_pos_ind] = F.sigmoid(
                    paddle.squeeze(
                        tea_iou, axis=-1)[loc_pos_ind])

        return cls_mask, loc_mask, \
            cls_targets, loc_targets, iou_targets

    def forward(self, student_prediction, teacher_prediction):
        stu_cls_lvl, stu_loc_lvl, stu_iou_lvl = student_prediction
        tea_cls_lvl, tea_loc_lvl, tea_iou_lvl, self.fpn_stride = teacher_prediction

        # H and W of level (used for aggregating targets)
        self.lvl_hw = []
        for t in tea_cls_lvl:
            _, _, H, W = t.shape
            self.lvl_hw.append([H, W])

        # levels to images
        stu_cls_img = levels_to_images(stu_cls_lvl)
        stu_loc_img = levels_to_images(stu_loc_lvl)
        stu_iou_img = levels_to_images(stu_iou_lvl)
        tea_cls_img = levels_to_images(tea_cls_lvl)
        tea_loc_img = levels_to_images(tea_loc_lvl)
        tea_iou_img = levels_to_images(tea_iou_lvl)

        with paddle.no_grad():
            cls_mask, loc_mask, \
                cls_targets, loc_targets, iou_targets = multi_apply(
                self.get_targets_per_img,
                tea_cls_img,
                tea_loc_img,
                tea_iou_img,
                stu_cls_img,
                stu_loc_img,
                stu_iou_img
            )

        # flatten preditction
        stu_cls = paddle.concat(stu_cls_img, axis=0)
        stu_loc = paddle.concat(stu_loc_img, axis=0)
        stu_iou = paddle.concat(stu_iou_img, axis=0)

        # flatten targets
        cls_mask = paddle.concat(cls_mask, axis=0)
        loc_mask = paddle.concat(loc_mask, axis=0)
        cls_targets = paddle.concat(cls_targets, axis=0)
        loc_targets = paddle.concat(loc_targets, axis=0)
        iou_targets = paddle.concat(iou_targets, axis=0)

        # Perform CRF post-processing
        emissions = F.softmax(stu_cls, axis=-1)  # Get emission scores
        _, new_labels = self.crf_post_process(stu_loc, cls_targets, emissions)
        # print(new_labels)
        # print(cls_targets)
        # print(stu_iou)
        cls_targets = new_labels
        # print(cls_targets)
        loss_cls = F.cross_entropy(stu_cls, cls_targets.argmax(axis=-1), ignore_index=-1)
        ### Training Weights and avg factor
        # find positives
        cls_pos_ind = paddle.nonzero(cls_mask > 0.).squeeze(axis=-1)
        loc_pos_ind = paddle.nonzero(loc_mask > 0.).squeeze(axis=-1)
        # cls weight
        cls_sample_weights = paddle.ones([cls_targets.shape[0]])
        cls_avg_factor = paddle.max(cls_targets[cls_pos_ind],
                                    axis=-1).sum().item()
        # loc weight
        loc_sample_weights = paddle.max(cls_targets[loc_pos_ind], axis=-1)
        loc_avg_factor = loc_sample_weights.sum().item()
        # iou weight
        iou_sample_weights = paddle.ones([loc_pos_ind.shape[0]])
        iou_avg_factor = loc_pos_ind.shape[0]

        ### unsupervised loss
        # cls loss

        # loss_cls = self.quality_focal_loss(
        #     stu_cls,
        #     cls_targets,
        #     quality=stu_iou,
        #     weights=cls_sample_weights,
        #     avg_factor=cls_avg_factor) * self.cls_weight
        # iou loss
        pos_stu_iou = paddle.squeeze(stu_iou, axis=-1)[loc_pos_ind]
        pos_iou_targets = iou_targets[loc_pos_ind]
        loss_iou = F.binary_cross_entropy(
            F.sigmoid(pos_stu_iou), pos_iou_targets,
            reduction='none') * iou_sample_weights
        loss_iou = loss_iou.sum() / iou_avg_factor * self.iou_weight
        # box loss
        pos_stu_loc = stu_loc[loc_pos_ind]
        pos_loc_targets = loc_targets[loc_pos_ind]

        loss_box = self.iou_loss(
            pos_stu_loc,
            pos_loc_targets,
            weights=loc_sample_weights,
            avg_factor=loc_avg_factor)
        loss_box = loss_box * self.reg_weight

        loss_all = {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_iou": loss_iou,
        }
        return loss_all
