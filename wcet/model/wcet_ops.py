from wcet.qlinearadd.wcet_qlinearadd import qlinearadd_wcet_analysis
from wcet.qlinearconv.wcet_qlinearconv import qlinearconv_wcet_analysis
from wcet.qlinearmatmul.wcet_qlinearmatmul import qlinearmatmul_wcet_analysis
from wcet.qlinearmul.wcet_qlinearmul import qlinearmul_wcet_analysis
from wcet.qlinearprelu.wcet_qlinearprelu import qlinearprelu_wcet_analysis
from wcet.qlinearrelu.wcet_qlinearrelu import qlinearrelu_wcet_analysis
from wcet.qlinearsigmoid.wcet_qlinearsigmoid import qlinearsigmoid_wcet_analysis
from wcet.qlinearsub.wcet_qlinearsub import qlinearsub_wcet_analysis
from wcet.qlineartanh.wcet_qlineartanh import qlineartanh_wcet_analysis
from wcet.transpose.wcet_transpose import transpose_wcet_analysis
from wcet.unsqueeze.wcet_unsqueeze import unsqueeze_wcet_analysis

def _test_qlinearadd():
    qlinearadd_wcet_analysis(name="qlinearadd-10", size=10, acctype="int32_t")
    qlinearadd_wcet_analysis(name="qlinearadd-100", size=100, acctype="int32_t")
    qlinearadd_wcet_analysis(name="qlinearadd-1000", size=1000, acctype="int32_t")

def _test_qlinearconv():
    qlinearconv_wcet_analysis(name="qlinearconv-10", KS=1, CIN=10, LIN=5, COUT=10, PAD=0, DIL=1, STR=1, Q=15, acctype="int32_t")
    qlinearconv_wcet_analysis(name="qlinearconv-100", KS=1, CIN=100, LIN=50, COUT=100, PAD=0, DIL=1, STR=1, Q=15, acctype="int32_t")

def _test_qlinearmatmul():
    qlinearmatmul_wcet_analysis(name="qlinearmatmul-10", M=10, K=10, N=1, Q=15, acctype="int32_t")
    qlinearmatmul_wcet_analysis(name="qlinearmatmul-100", M=100, K=100, N=1, Q=15, acctype="int32_t")
    qlinearmatmul_wcet_analysis(name="qlinearmatmul-1000", M=1000, K=1000, N=1, Q=15, acctype="int32_t")

def _test_qlinearmul():
    qlinearmul_wcet_analysis(name="qlinearmul-10", size=10, Q=15, acctype="int32_t")
    qlinearmul_wcet_analysis(name="qlinearmul-100", size=100, Q=15, acctype="int32_t")
    qlinearmul_wcet_analysis(name="qlinearmul-1000", size=1000, Q=15, acctype="int32_t")

def _test_qlinearprelu():
    qlinearprelu_wcet_analysis(name="qlinearprelu-10", size=10, Q=15, acctype="int32_t")
    qlinearprelu_wcet_analysis(name="qlinearprelu-100", size=100, Q=15, acctype="int32_t")
    qlinearprelu_wcet_analysis(name="qlinearprelu-1000", size=1000, Q=15, acctype="int32_t")

def _test_qlinearrelu():
    qlinearrelu_wcet_analysis(name="qlinearrelu-10", size=10, acctype="int32_t")
    qlinearrelu_wcet_analysis(name="qlinearrelu-100", size=100, acctype="int32_t")
    qlinearrelu_wcet_analysis(name="qlinearrelu-1000", size=1000, acctype="int32_t")

def _test_qlinearsigmoid():
    qlinearsigmoid_wcet_analysis(name="qlinearsigmoid-10", size=10, acctype="int32_t")
    qlinearsigmoid_wcet_analysis(name="qlinearsigmoid-100", size=100, acctype="int32_t")
    qlinearsigmoid_wcet_analysis(name="qlinearsigmoid-1000", size=1000, acctype="int32_t")

def _test_qlinearsub():
    qlinearsub_wcet_analysis(name="qlinearsub-10", size=10, acctype="int32_t")
    qlinearsub_wcet_analysis(name="qlinearsub-100", size=100, acctype="int32_t")
    qlinearsub_wcet_analysis(name="qlinearsub-1000", size=1000, acctype="int32_t")

def _test_qlineartanh():
    qlineartanh_wcet_analysis(name="qlineartanh-10", size=10, acctype="int32_t")
    qlineartanh_wcet_analysis(name="qlineartanh-100", size=100, acctype="int32_t")
    qlineartanh_wcet_analysis(name="qlineartanh-1000", size=1000, acctype="int32_t")

def _test_transpose():
    transpose_wcet_analysis(name="transpose-10", cols=10, rows=1)
    transpose_wcet_analysis(name="transpose-100", cols=100, rows=1)
    transpose_wcet_analysis(name="transpose-1000", cols=1000, rows=1)

def _test_unsqueeze():
    unsqueeze_wcet_analysis(name="unsqueeze-10", size=10)
    unsqueeze_wcet_analysis(name="unsqueeze-100", size=100)
    unsqueeze_wcet_analysis(name="unsqueeze-1000", size=1000)


if __name__ == '__main__':
    _test_qlinearadd()
    _test_qlinearconv()
    _test_qlinearmatmul()
    _test_qlinearmul()
    _test_qlinearprelu()
    _test_qlinearrelu()
    _test_qlinearsigmoid()
    _test_qlinearsub()
    _test_qlineartanh()
    _test_transpose()
    _test_unsqueeze()
