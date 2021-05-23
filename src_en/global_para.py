
class Paras:

    COURSE_LABEL_SIZE = None
    FINE_2_COURSE = None
    IX_a = None
    IX_t = None
    IX_s = None
    IX_a_out = None
    IX_p_out = None

    @staticmethod
    def init():
        print("Paras.init started...")
        Paras.COURSE_LABEL_SIZE = 6
        Paras.FINE_2_COURSE = {}
        Paras.FINE_2_COURSE[()] = 0

        Paras.FINE_2_COURSE[("NP", )] = 1
        Paras.FINE_2_COURSE[("WHNP",)] = 1
        Paras.FINE_2_COURSE[("NAC",)] = 1
        Paras.FINE_2_COURSE[("NX",)] = 1

        Paras.FINE_2_COURSE[("S", )] = 2
        Paras.FINE_2_COURSE[("SINV",)] = 2
        Paras.FINE_2_COURSE[("SBAR", )] = 2
        Paras.FINE_2_COURSE[("SBARQ",)] = 2
        Paras.FINE_2_COURSE[("SQ", )] = 2
        Paras.FINE_2_COURSE[("VP",)] = 2

        Paras.FINE_2_COURSE[("ADJP",)] = 3
        Paras.FINE_2_COURSE[("ADVP",)] = 3
        Paras.FINE_2_COURSE[("PP",)] = 3
        Paras.FINE_2_COURSE[("WHADJP",)] = 3
        Paras.FINE_2_COURSE[("WHADVP",)] = 3
        Paras.FINE_2_COURSE[("WHPP",)] = 3
        Paras.FINE_2_COURSE[("PRN",)] = 3
        Paras.FINE_2_COURSE[("QP",)] = 3
        Paras.FINE_2_COURSE[("RRC",)] = 3
        Paras.FINE_2_COURSE[("UCP",)] = 3
        Paras.FINE_2_COURSE[("LST",)] = 3

        Paras.FINE_2_COURSE[("INTJ",)] = 4
        Paras.FINE_2_COURSE[("PRT",)] = 4
        Paras.FINE_2_COURSE[("CONJP",)] = 4

        Paras.FINE_2_COURSE[("X",)] = 5
        Paras.FINE_2_COURSE[("FRAG",)] = 5

        Paras.IX_a = {}
        Paras.IX_t = {}
        Paras.IX_s = {}
        Paras.IX_a_out = {}
        Paras.IX_p_out = {}
        for span_len in range(1, 500):
            Paras.IX_a[span_len] = [[(0, j, i) for i in range(0, span_len - 1)] for j in range(1, span_len)]
            Paras.IX_t[span_len] = [[(i, j, span_len) for i in range(0, span_len - 1)] for j in range(1, span_len)]
            Paras.IX_s[span_len] = [[(j, span_len) for i in range(0, span_len - 1)] for j in range(1, span_len)]
            Paras.IX_a_out[span_len] = [(0, span_len, j) for j in range(0, span_len)]
            Paras.IX_p_out[span_len] = [(0, span_len, j) for j in range(1, span_len)]

        print("Paras.init finished...")


if __name__ == '__main__':
    Paras.init()

