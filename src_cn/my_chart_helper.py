import numpy as np
import torch
import trees
from global_para import *
from myutil import *


def my_decode(seq_feature, is_train, gold, label_vocab, course_label_ix, score_model, label_model, sent):

    with torch.no_grad():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        feat_dim = seq_feature.size(1)
        sent_len = seq_feature.size(0)-1
        label_size = Paras.COURSE_LABEL_SIZE

        if is_train:
            gold_course = []
            for span in gold:
                i, j, lbl = span
                lbl_course = label_vocab.value(lbl)
                if lbl_course is ():
                    lbl_course = Paras.FINE_2_COURSE.get(lbl_course)
                else:
                    lbl_course = Paras.FINE_2_COURSE.get((lbl_course[0],))
                gold_course.append((i, j, lbl_course))

        # step 1 calculate t_score
        span_features = (torch.unsqueeze(seq_feature, 0)
                         - torch.unsqueeze(seq_feature, 1))
        # a = span_features.unsqueeze(2).expand(sent_len+1, sent_len+1, sent_len+1, feat_dim)
        # b = span_features.unsqueeze(0).expand(sent_len+1, sent_len+1, sent_len+1, feat_dim)
        # t_score_feat = torch.cat([a, b], 3)

        e_score = label_model(span_features)
        # t_score = torch.zeros(sent_len+1, sent_len+1, sent_len+1, label_size*label_size)
        t_score = None
        for i in range(0, sent_len+1):
            t_score_i = score_model(torch.cat([span_features[i].unsqueeze(1).expand(sent_len+1, sent_len+1, feat_dim),
                                   span_features], 2))
            if i == 0:
                t_score = t_score_i.unsqueeze(0)
            else:
                t_score = torch.cat([t_score, t_score_i.unsqueeze(0)], 0)
            del t_score_i

        if is_train:
            for span in gold:
                i, j, lbl = span
                e_score[i][j][lbl] -= 1.
            gold_index = my_gen_clique_spans(gold_course)
            for span_info in gold_index:
                i, j, k, lbl_l, lbl_r = span_info
                t_score[i][j][k][lbl_r*label_size+lbl_l] -= 1.
            e_score += 1.
            t_score += 1.

        # step 2 calculate alpha
        s_score = -np.inf*torch.ones(sent_len+1, sent_len+1)
        alpha = -np.inf*torch.ones(sent_len+1, sent_len+1, sent_len+1, label_size)

        p_ix_lbl = -1*torch.ones(sent_len+1, sent_len+1, sent_len+1, label_size, dtype=torch.short)
        p_ix_spn = -1*torch.ones(sent_len+1, sent_len+1, sent_len+1, label_size, dtype=torch.short)
        c_ix_lbl = -1*torch.ones(sent_len+1, sent_len+1, dtype=torch.short)
        c_ix_spn = -1*torch.ones(sent_len+1, sent_len+1, dtype=torch.short)

        e_score_max = []
        e_score_maxlbl = []
        for i in range(0, label_size):
            select_ix = torch.cuda.LongTensor(course_label_ix[i]).view\
                (1, 1, len(course_label_ix[i])).expand(sent_len+1, sent_len+1, len(course_label_ix[i]))
            e_score_select = torch.gather(e_score, 2, select_ix)
            e_score_s, e_score_ix_s = torch.max(e_score_select, 2)
            e_score_max.append(e_score_s.unsqueeze(2))
            e_score_maxlbl.append(e_score_ix_s.unsqueeze(2))

        del e_score
        e_score = torch.cat(e_score_max, 2)
        e_score_ix = torch.cat(e_score_maxlbl, 2)

        for span_len in range(1, sent_len+1):
            span_num = sent_len - span_len + 1
            if span_len == 1:
                ix_put1 = torch.cuda.LongTensor([(0+k, 1+k, 0+k) for k in range(0, span_num)])
                ix_lbl = torch.cuda.LongTensor([(0+k, 1+k) for k in range(0, span_num)])
                e_score_new = my_index_select(e_score, ix_lbl)
                my_index_put(alpha, ix_put1, e_score_new)
                ix_put2 = torch.cuda.LongTensor([(0+k, 1+k) for k in range(0, span_num)])
                my_index_put(s_score, ix_put2, torch.zeros(span_num, dtype=torch.float))
            else:
                # step 1 给非0维赋值
                # a_ix_cal = torch.cuda.LongTensor([[[(0+k, j+k, i+k) for i in range(0, span_len-1)] for j in range(1, span_len)] for k in range(0, span_num)])
                # t_ix_cal = torch.cuda.LongTensor([[[(i+k, j+k, span_len+k) for i in range(0, span_len-1)] for j in range(1, span_len)] for k in range(0, span_num)])
                # s_ix_cal = torch.cuda.LongTensor([[[(j+k, span_len+k) for i in range(0, span_len-1)] for j in range(1, span_len)] for k in range(0, span_num)])
                # e_ix_cal = torch.cuda.LongTensor([[[(j+k, span_len+k) for i in range(0, span_len-1)] for j in range(1, span_len)] for k in range(0, span_num)])

                a_ix_cal1 = torch.cuda.LongTensor(Paras.IX_a.get(span_len)).unsqueeze(0).expand(span_num, span_len - 1,
                                                                                                span_len - 1, 3)
                a_ix_cal2 = torch.cuda.LongTensor([k for k in range(0, span_num)]).view(span_num, 1).expand(span_num, (
                            span_len - 1) * (span_len - 1) * 3).reshape(span_num, span_len - 1, span_len - 1, 3)
                a_ix_cal = a_ix_cal1 + a_ix_cal2
                del a_ix_cal1
                del a_ix_cal2

                t_ix_cal1 = torch.cuda.LongTensor(Paras.IX_t.get(span_len)).unsqueeze(0).expand(span_num, span_len - 1,
                                                                                                span_len - 1, 3)
                t_ix_cal2 = torch.cuda.LongTensor([k for k in range(0, span_num)]).view(span_num, 1).expand(span_num, (
                            span_len - 1) * (span_len - 1) * 3).reshape(span_num, span_len - 1, span_len - 1, 3)
                t_ix_cal = t_ix_cal1 + t_ix_cal2
                del t_ix_cal1
                del t_ix_cal2

                s_ix_cal1 = torch.cuda.LongTensor(Paras.IX_s.get(span_len)).unsqueeze(0).expand(span_num, span_len - 1,
                                                                                                span_len - 1, 2)
                s_ix_cal2 = torch.cuda.LongTensor([k for k in range(0, span_num)]).view(span_num, 1).expand(span_num, (
                            span_len - 1) * (span_len - 1) * 2).reshape(span_num, span_len - 1, span_len - 1, 2)
                s_ix_cal = s_ix_cal1 + s_ix_cal2
                del s_ix_cal1
                del s_ix_cal2

                a = my_index_select(alpha, a_ix_cal)
                t = my_index_select(t_score, t_ix_cal)
                s = my_index_select(s_score, s_ix_cal)
                e = my_index_select(e_score, s_ix_cal)
                del a_ix_cal
                del t_ix_cal
                del s_ix_cal

                a_new0 = a.repeat(1, 1, 1, label_size)+t+s.unsqueeze(3).repeat(1, 1, 1, label_size*label_size)+\
                         e.unsqueeze(4).expand(t.size(0), t.size(1), t.size(2), label_size, label_size).reshape(t.size(0), t.size(1), t.size(2), label_size*label_size)
                del a
                del t
                del s
                del e

                a_new1, tmp_ix1 = torch.max(a_new0.reshape(a_new0.size(0), a_new0.size(1), a_new0.size(2), label_size, label_size), 4)  # tmp_ix1:k, j, i, label
                del a_new0

                a_new2, p_ix_spn_new = torch.max(a_new1.transpose(2, 3), 3)  # tmp_sp: k, j, label
                p_ix_lbl_new = torch.gather(tmp_ix1.transpose(2, 3), 3, p_ix_spn_new.unsqueeze(3)).squeeze(3)
                del a_new1

                # step 2 给0维赋值
                a0_new, tmp_ix3 = torch.max(a_new2, 2)
                a0_new, c_ix_spn_new = torch.max(a0_new, 1)
                c_ix_lbl_new = torch.gather(tmp_ix3, 1, c_ix_spn_new.unsqueeze(1)).squeeze(1)

                s_new = a0_new

                ix_lbl = torch.cuda.LongTensor([(0 + k, span_len + k) for k in range(0, span_num)])
                e_score_new = my_index_select(e_score, ix_lbl)
                a0_new = a0_new.view(span_num, 1).expand(span_num, label_size)+e_score_new

                a0_new = a0_new.view(span_num, 1, label_size)
                a_new = torch.cat([a0_new, a_new2], 1)
                del a_new2
                del a0_new

                # step 3 写入结果

                # ix_alpha = torch.cuda.LongTensor([(i, i+span_len, i+j) for i in range(0, span_num) for j in range(0, span_len)])
                ix_alpha1 = torch.cuda.LongTensor(Paras.IX_a_out.get(span_len)).unsqueeze(0).expand(span_num, span_len, 3).reshape(span_num*span_len, 3)
                ix_alpha2 = torch.cuda.LongTensor([i for i in range(0, span_num)]).view(span_num, 1).expand(span_num, span_len*3).reshape(span_num, span_len, 3).reshape(span_num*span_len, 3)
                ix_alpha = ix_alpha1+ix_alpha2
                del ix_alpha1
                del ix_alpha2
                my_index_put(alpha, ix_alpha, a_new.reshape(span_num*span_len, label_size))
                del ix_alpha
                # ix_p = torch.cuda.LongTensor([(i, i + span_len, i+j) for i in range(0, span_num) for j in range(1, span_len)])
                ix_p1 = torch.cuda.LongTensor(Paras.IX_p_out.get(span_len)).unsqueeze(0).expand(span_num, span_len-1, 3).reshape(span_num*(span_len-1), 3)
                ix_p2 = torch.cuda.LongTensor([i for i in range(0, span_num)]).view(span_num, 1).expand(span_num, (span_len-1)*3).reshape(span_num, span_len-1, 3).reshape(span_num*(span_len-1), 3)
                ix_p = ix_p1+ix_p2
                del ix_p1
                del ix_p2
                my_index_put(p_ix_lbl, ix_p, p_ix_lbl_new.short().reshape(span_num*(span_len-1), label_size))
                my_index_put(p_ix_spn, ix_p, p_ix_spn_new.short().reshape(span_num*(span_len-1), label_size))
                del ix_p
                ix_score = torch.cuda.LongTensor([(i, i+span_len) for i in range(0, span_num)])
                my_index_put(s_score, ix_score, s_new)
                ix_c = torch.cuda.LongTensor([(i, i+span_len) for i in range(0, span_num)])
                my_index_put(c_ix_lbl, ix_c, c_ix_lbl_new.short())
                my_index_put(c_ix_spn, ix_c, c_ix_spn_new.short())
                del a_new
                del p_ix_spn_new
                del p_ix_lbl_new
                del s_new
                del c_ix_spn_new
                del c_ix_lbl_new

        span_list = []
        tmp_list = [(0, span_len, torch.max(e_score[0][sent_len], 0)[1])]
        while len(tmp_list) > 0:
            span = tmp_list.pop()
            span_list.append(span)
            ix_c_lbl = int(c_ix_lbl[(span[0], span[1])])
            ix_c_spn = int(c_ix_spn[(span[0], span[1])])  # count from 1
            if ix_c_spn >= 0:
                ix_c_spn += 1
            span_s = span[0]
            span_e = span[1]
            while ix_c_spn >= 0:
                ix_p_lbl = int(p_ix_lbl[span_s, span_e, span_s+ix_c_spn, ix_c_lbl])
                ix_p_spn = int(p_ix_spn[span_s, span_e, span_s+ix_c_spn, ix_c_lbl])
                tmp_list.append((span_s+ix_c_spn, span_e, ix_c_lbl))
                span_e = span_s+ix_c_spn
                ix_c_lbl = ix_p_lbl
                ix_c_spn = ix_p_spn

        max_s = s_score[0][sent_len]+torch.max(e_score[0][sent_len], 0)[0]

        del s_score
        del alpha
        del p_ix_lbl
        del p_ix_spn
        del c_ix_lbl
        del c_ix_spn

    if is_train:
        pred_score = 0.
        span_features = (torch.unsqueeze(seq_feature, 0)
                         - torch.unsqueeze(seq_feature, 1))
        clique_spans = my_gen_clique_spans(span_list)
        for clique_span in clique_spans:
            i, j, k, lbl_ij, lbl_jk = clique_span
            clique_score = score_model(torch.cat([span_features[i][j], span_features[j][k]], 0))[lbl_jk*label_size+lbl_ij]
            if pred_score is None:
                pred_score = clique_score
            else:
                pred_score = pred_score+clique_score
        for span in span_list:
            i, j, lbl = span
            pred_score = pred_score+label_model(span_features[i][j])[course_label_ix[lbl][e_score_ix[i][j][lbl]]]
        margin = float((max_s - pred_score).clone().detach().cpu().numpy())

        gold_score = 0.
        clique_spans = my_gen_clique_spans(gold_course)
        for clique_span in clique_spans:
            i, j, k, lbl_ij, lbl_jk = clique_span
            clique_score = score_model(torch.cat([span_features[i][j], span_features[j][k]], 0))[lbl_jk*label_size+lbl_ij]
            if gold_score is None:
                gold_score = clique_score
            else:
                gold_score = gold_score+clique_score
        for span in gold:
            i, j, lbl = span
            gold_score = gold_score+label_model(span_features[i][j])[lbl]

        loss = torch.relu(pred_score-gold_score+margin)
        return None, loss
    else:
        label_list = []
        for span in span_list:
            i, j, lbl = span
            label_list.append(course_label_ix[lbl][e_score_ix[i][j][lbl]])

        idx = -1
        def make_tree():
            nonlocal idx
            idx += 1
            i, j, _ = span_list[idx]
            lbl = label_list[idx]
            label = label_vocab.value(lbl)
            if (i + 1) >= j:
                tag, word = sent[i]
                tree = trees.LeafParseNode(int(i), tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], j
            else:
                first_trees, jj = make_tree()
                children = first_trees
                while jj < j:
                    right_trees, jj = make_tree()
                    children = children+right_trees
                if label:
                    return [trees.InternalParseNode(label, children)], j
                else:
                    return [trees.InternalParseNode(("None", ), children)], j

        tree, _ = make_tree()

        return tree[0], None



















