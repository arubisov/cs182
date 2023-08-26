import numpy as np
import os

model_out_to_list = lambda t,m: \
    (lambda t: list((ti.detach().cpu().numpy().tolist() for ti in t)) if isinstance(t,tuple)\
                                 else t.detach().cpu().numpy().tolist())(m(*t))
validate_to_array = lambda s, i, f, r, multi=False: \
    (lambda o: np.save(os.path.join(r,'submission_logs',f),o,allow_pickle=True))(
      (
        (lambda t: t[0] if len(t) == 1 else t)(
            tuple((s(*i_i) for i_i in ((i,) if not multi else i)))
        ),
        os.makedirs(os.path.join(r,'submission_logs'),exist_ok=True)
      )[0]
        
    )
