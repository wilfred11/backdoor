from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD


do=2

def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup


if do==2:
    causal_model = DiscreteBayesianNetwork([('male', 'drug'), ('male', 'recovery'), ('drug', 'recovery')])
    causal_model.get_random_cpds(n_states=2, inplace=True, seed=1)

    viz = causal_model.to_graphviz()
    viz.draw('out/drug_model.png', prog='dot')

    fnames=["male", "m-d", "m-d-r"]
    for cpt, fname in zip(causal_model.get_cpds(),fnames):
        print_full(cpt)
        cpt.to_csv(filename="out/"+fname+'.csv')

    causal_inference = CausalInference(causal_model)

    do_drug_1 = causal_inference.query(variables=["recovery"], do={"drug": 1}, show_progress=False)
    do_drug_0 = causal_inference.query(variables=["recovery"], do={"drug": 0}, show_progress=False)

    print(f"If the drug is taken by everyone p(recovery)={do_drug_1.get_value(recovery=1):.4}")
    print(f"If the drug is not taken by anyone p(recovery)={do_drug_0.get_value(recovery=1):.4}")

    print(
        f"\nThe improvement in recovery rate by everyone taking the drug is {do_drug_1.get_value(recovery=1) - do_drug_0.get_value(recovery=1):.1%}")

    d_effect = (0.649*0.3013)+(0.35*0.3081)
    nd_effect = (0.649*0.8516)+(0.35*0.3785)
    print("drugs effect on recover")
    print(d_effect)
    print("no drugs effect on recovery")
    print(nd_effect)
    print(d_effect-nd_effect)






