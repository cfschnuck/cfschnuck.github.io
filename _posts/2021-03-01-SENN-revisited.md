---
title: "Self-Explaining Neural Networks revisited"
img_dir: "/assets/images/posts/2020/senn/"
header:
  teaser: "/assets/images/posts/2020/senn/teaser.png"
  og_image: "/assets/images/posts/2020/senn/teaser.png"
toc: true
read_time: true
usemathjax: true
---

Today's machine learning methods are known for their ability to produce high performing models from data alone. However, the most common approaches in machine learning are often far too complex for humans to understand, hence they are often referred to as *black-box* models. For humans, it is impossible to understand how these models arrived at their decisions. At first glance, this does not seem to be a problem, as these models perform very well in almost all application areas, sometimes even surpassing human performance. Therefore, it is not surprising that these models have been adopted  in  a  wide  range  of  real-world  applications, especially  in  increasingly  critical  domains  such  as  healthcare,  criminal  justice  and  autonomous  driving. However, when taking a deeper look, it becomes clear that all of these applications have a direct impact on our lives and can be harmful to our society if not designed and engineered correctly, with considerations to fairness. Consequently, demand for interpretability of complex machine learning models  has  increased  over  recent  years.

## Achieving Interpretability
Two different paradigms on achieving interpretability of machine learning  models  can  be  found  in  the  literature: post-hoc explanation  methods  and  intrinsically  interpretable  models.

Post-hoc explanation methods aim to provide information on *black-box* models after training and do not pose any constraints or regularizations enforcing interpretability on the model of interest. These methods can be divided into gradient or reverse propagation methods that use at least partial information given by the primary model, and complete *black-box* methods that only use local input-output behavior for explanations.

Conversely, an intrinsically interpretable approach aims to make a model interpretable by its formulation and architecture and thus takes into account interpretability already during training. A first and natural approach to achieve model interpretability is to restrict oneself to the use of simple and often close to linear models that can be understood by humans on a global level. Other approaches aim to induce sparsity in the context of a neural network. Lastly and most relevant for the approach taken here, we want to mention explanation generating models. These models produce explanations which are at the same time used for predictions and thus aim to be intrinsically interpretable by network architecture. 

## Self-Explaining Neural Networks (SENN)

![senn]({{page.img_dir}}teaser.png "SENN")

In this work, we use an explanation generating architecture called [Self-Explaining Neural Network *SENN* from David Alvarez-Melis and Tommi Jaakkola](https://arxiv.org/abs/1806.07538) [[1]](#1).

A *SENN* consists of two components: a conceptizer $$h(x)$$ 