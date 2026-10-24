"""The model ids a test means when it says "the coding launch" or "the second model".

A launch is filed under the model it loads: it is the key of the Compose service, of the manifest
entry, of the observation record, and of an operator's alias. So a test that cares about one launch
names the model rather than a slot, and says it once here instead of re-deriving it from a profile
file in every module.
"""

CODING = 'qwen3.8-27b-q8_0-tensor-next'
CODING_Q6K = 'qwen3.8-27b-ud-q6_k_xl-tensor-next'
EVERYDAY = 'qwen3.8-27b-abliterated-q6_k'
# The models of the Strata configurations: one on every card, and one on a pair.
STRATA_SOLO = 'qwen3.8-flash-next-ud-iq4_xs'
STRATA_PAIR = 'qwen3.8-flash-next-iq3_s'
STRATA_DAISY = 'qwen3.8-flash-next-iq3_xxs'
