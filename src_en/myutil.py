import torch


def my_index_put(data, ixs, value):
    ix_put = ixs.split(1, -1)
    for i in range(0, len(ix_put)):
        ix_put[i].squeeze_(len(ixs.size()) - 1)
    data[ix_put] = value


def my_index_select(data, ixs):
    if len(data.size()) < ixs.size(-1):
        print("ix select error...")
        return None
    # step 1 把ixs转成flat
    # a = ixs.split(1, len(ixs.size()) - 1)
    a = ixs.split(1, -1)
    factor = 1
    ixs_flat_inside = a[-1]
    for i in reversed(range(1, len(a))):
        factor = factor * data.size(i)
        ixs_flat_inside = ixs_flat_inside + factor * a[i - 1]
    ixs_flat_outside = ixs_flat_inside.reshape(-1)
    # step 2 把data 转成flat
    dim_tmp = 1
    for i in range(0, ixs.size(-1)):
        dim_tmp = dim_tmp * data.size(i)
    t_flat = data.reshape(dim_tmp, -1)
    # step 3 index_select
    result = torch.index_select(t_flat, 0, ixs_flat_outside)
    # step 4 reshape back
    shape_size = list(ixs.size()[0:len(ixs.size())-1] + data.size()[ixs.size(-1):])
    result = result.reshape(shape_size)
    return result


def my_gen_clique_spans(span_list):
    start_2_goldspan = {}
    end_2_goldspan = {}
    max_right = -1
    for span in span_list:
        left, right, label = span
        if right>max_right:
            max_right = right
        if start_2_goldspan.get(left) is None:
            start_2_goldspan[left] = []
        if end_2_goldspan.get(right) is None:
            end_2_goldspan[right] = []
        start_2_goldspan.get(left).append((left, right, label))
        end_2_goldspan.get(right).append((left, right, label))

    gold_index = []

    for i in range(1, max_right):
        if (start_2_goldspan.get(i) is not None) and (end_2_goldspan.get(i) is not None):
            spans_start = start_2_goldspan.get(i)
            spans_end = end_2_goldspan.get(i)
            max_start_spn = None
            max_start_lbl = None
            tmp_len = 0
            for span in spans_start:
                left, right, label = span
                if right - left > tmp_len:
                    tmp_len = right - left
                    max_start_spn = (left, right)
                    max_start_lbl = label
            max_end_spn = None
            max_end_lbl = None
            tmp_len = 0
            for span in spans_end:
                left, right, label = span
                if right - left > tmp_len:
                    tmp_len = right - left
                    max_end_spn = (left, right)
                    max_end_lbl = label
            gold_index.append((max_end_spn[0], max_start_spn[0], max_start_spn[1], max_end_lbl, max_start_lbl))
    return gold_index



