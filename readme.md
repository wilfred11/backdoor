## Backdoor 

When an effect is confounded, one can use pgmpy to calculate the effect of a drug on recovery. Male is the confounder that in the ideal case should influence 'drug' use.

![drug_model.png](out/drug_model.png)

### Code

The imports

`from pgmpy.inference import CausalInference`

`from pgmpy.models import DiscreteBayesianNetwork`

`from pgmpy.factors.discrete.CPD import TabularCPD`

To find the effect of taking a drug on recovery, one first needs to create a causal model.

`causal_model = DiscreteBayesianNetwork([('male', 'drug'), ('male', 'recovery'), ('drug', 'recovery')])`

`causal_model.get_random_cpds(n_states=2, inplace=True, seed=1)`

To create a visual representation of the causal graph

`viz = causal_model.to_graphviz()`

`viz.draw('out/drug_model.png', prog='dot')`

To infer the effect of drug use on recovery

`causal_inference = CausalInference(causal_model)`

`do_drug_1 = causal_inference.query(variables=["recovery"], do={"drug": 1}, show_progress=False)`

`do_drug_0 = causal_inference.query(variables=["recovery"], do={"drug": 0}, show_progress=False)`

To retrieve the values

    `print(f"If the drug is taken by everyone p(recovery)={do_drug_1.get_value(recovery=1):.4}")`

    `print(f"If the drug is not taken by anyone p(recovery)={do_drug_0.get_value(recovery=1):.4}")`

### Calculate by hand

![male.png](out/male.png)

![m-d-r.png](out/m-d-r.png)

Calculate effect when using drug.

$$P(recovery=1|do(drug=1))= \sum_{male=\{0,1\}}  P(recovery=1|drug=1,male=male) * P(male=male)$$

Calculate effect when not using drug.

$$P(recovery=1|do(drug=0))= \sum_{male=\{0,1\}}  P(recovery=1|drug=0,male=male) * P(male=male)$$


`P(recovery=1|do(drug=1))=((0.649*0.3013)+(0.35*0.3081))=0.3033`

`P(recovery=1|do(drug=0))=((0.649*0.8516)+(0.35*0.3785))=0.6851`

`P(recovery=1|do(drug=1))-P(recovery=1|do(drug=0))=-0.38`

