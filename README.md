# 🔮 mī lyte: Alignment, feasibility, and optimization of an LLM emotion regulation intervention to deliver just-in-time mindfulness skills and PrEP education
[Simone J. Skeen](https://simoneskeen.xyz/)

<p align="center">
  <img src= "images/sample.png" alt="A web browser screenshot showing a chatbot interface. The sidebar reads: mi lyte: Accessible mindfulness skills for everyday life. mHEAL: Mindfulness for Health Equity Lab. A disclaimer reads: mī lyte is a tool to provide knowledge on stress and resilience, recommending evidence-based informal mindfulness practices to build into your everyday life. mī lyte is not a replacement for in-person therapy. It does not provide medical or psychiatric advice. The user query reads: I have too much to deal with today! I feel so overwhelmed I can't even start. The response reads: It sounds like you're in the midst of a stress spiral—overwhelm leads to frustration, which makes everything feel harder. Remember, this is temporary but can feel endless if we don’t pause. Here’s what helps: Mindful Breathing: Take 3 deep breaths. Inhale for 4 counts, hold for 4, exhale for 6. This grounds you and calms your nervous system. Acknowledge Your Feelings: It’s okay to feel overwhelmed—don’t fight it. Label what you’re feeling (e.g., “I’m stressed”) without judgment. Try this: Write down one thing you can do right now, no matter how small. Check it off when done. You’ve already succeeded! A prompt for additional dialogue reads: I could use some help with..." width="900" height=auto>
  <br>
  <em>Illustrative example: an evidence-based mī lyte recommendation delivered via the web app UI.</em>
</p>

A prototype developed by [Dr. Simone Skeen](https://sites.brown.edu/mheal/team/#simone) for [mHEAL: the Mindfulness for Health Equity Lab](https://sites.brown.edu/mheal/) at Brown University, grounded in _Mindfulness-Based Queer Resilience_ © [Dr. Shufang Sun](https://sites.brown.edu/mheal/team/#shufang). Uniquely, to our knowledge, mī lyte combines a mindfulness-based stress resilience (MBSR; System 1) and a PrEP education (System 2) intervention, leveraging the unique strengths of Transformer-based LLMs and traditional (pre-scripted) text message banks, respectively, in an Information-Motivation-Behavioral skills–grounded manner.[^1] 

<p align="center">
  <img src= "images/conceptual_model.jpg" alt="A conceptual model of boxes and arrows. Early life adversity is shown leading to chronic stress reactivity, which can lead to maladaptive coping, depression, hazardous alcohol use, illicit drug use, and condomless and/or serostatus-unaware sex. In contrast, chronic stress reactivity can be decoupled via mindfulness-based stress regulation, leading to emotion regulation, accurate risk perception, and PrEP knowledge and eventual PrEP uptake, aided by mi lyte System 1 and System 2." width="900" height=auto>
  <br>
  <em>Conceptual model: mindfulness facilitating emotion regulation (proximally) while driving IMB-consistent mechanisms of PrEP uptake (distally).</em>
</p>

### 🍂 System 1
Tapping a retrieval-augmented generation pipeline, user-queried stressors to mī lyte elicit MBSR skills recommendations to practice in real time. "Retrieval-augmented generation" refers to a system in which specialized knowledge is “retrieved” from a knowledge base – Dr. Sun’s internet mindfulness-based intervention manual (NCCIH #K23AT011173), in mī lyte – to “augment” a response “generated” by an LLM.[^2] This knowledge base can be personalized by the end user to incorporate individual resilience assets: inspiring song lyrics, aspects of faith and spirituality. Prioritizing safety, mī lyte System 1 will firmly but politely refuse queries that are outside of the context of its MBSR knowledge base. 

### 🍑 System 2
PrEP education encoded in a pre-scripted text message bank will be navigable to users via a deterministic dialog tree. Circumventing the probabilistic process by which LLMs generate responses, System 2 eliminates the possibility of “factuality hallucinations,” i.e. inaccurate medical information, addressing an urgent challenge in healthcare AI.[^3] 

[^1]: Dubov A, Altice FL, Fraenkel L. An Information–Motivation–Behavioral skills model of PrEP uptake. _AIDS Behav_. 2018;22(11):3603-3616. [doi:10.1007/s10461-018-2095-4](https://pubmed.ncbi.nlm.nih.gov/29557540/)
[^2]: Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. _arXiv_. Preprint posted online April 12, 2021. [doi:10.48550/arXiv.2005.11401](https://arxiv.org/abs/2005.11401)
[^3]: Obradovich N, Khalsa SS, Khan WU, et al. Opportunities and risks of large language models in psychiatry. _NPP—Digit Psychiatry Neurosci_. 2024;2(1). [doi:10.1038/s44277-024-00010-z](https://pubmed.ncbi.nlm.nih.gov/39554888/)


