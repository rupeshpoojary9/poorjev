"""Pluggable backends. M1 ships none; the local NLI backend lands in M2 and the
opt-in LLM backend in M5. A backend's whole job is to turn state + a primitive's
hypotheses into a vector of raw scores; the primitive does the rest.
"""
