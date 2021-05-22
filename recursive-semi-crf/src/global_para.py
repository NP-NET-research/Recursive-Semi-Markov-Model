
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
        Paras.FINE_2_COURSE[("IP", )] = 2
        Paras.FINE_2_COURSE[("VP",)] = 2
        Paras.FINE_2_COURSE[("FRAG",)] = 3
        Paras.FINE_2_COURSE[("UCP",)] = 3
        Paras.FINE_2_COURSE[("INTJ",)] = 3
        Paras.FINE_2_COURSE[("FLR",)] = 3
        Paras.FINE_2_COURSE[("INC",)] = 3
        Paras.FINE_2_COURSE[("DFL",)] = 3		
        Paras.FINE_2_COURSE[("ADJP",)] = 4
        Paras.FINE_2_COURSE[("ADVP",)] = 4
        Paras.FINE_2_COURSE[("PP",)] = 4
        Paras.FINE_2_COURSE[("DP",)] = 4
        Paras.FINE_2_COURSE[("QP",)] = 4
        Paras.FINE_2_COURSE[("LCP",)] = 4
        Paras.FINE_2_COURSE[("DNP",)] = 4
        Paras.FINE_2_COURSE[("CP",)] = 4
        Paras.FINE_2_COURSE[("DVP",)] = 4
        Paras.FINE_2_COURSE[("LST",)] = 4
        Paras.FINE_2_COURSE[("PRN",)] = 4
        Paras.FINE_2_COURSE[("CLP",)] = 4
        Paras.FINE_2_COURSE[("VCD",)] = 5
        Paras.FINE_2_COURSE[("VCP",)] = 5
        Paras.FINE_2_COURSE[("VNV",)] = 5
        Paras.FINE_2_COURSE[("VPT",)] = 5
        Paras.FINE_2_COURSE[("VRD",)] = 5
        Paras.FINE_2_COURSE[("VSB",)] = 5




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

