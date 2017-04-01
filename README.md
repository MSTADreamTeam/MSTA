# MSTA: Multi Strategy Trading Algorithm


Hybrid trading algorithm using a set of predictive methods to detect trends in market price time series. The set of predictions is then retreated through an ensemble method to provide a final prediction and a trading strategy is built on it.

## Main guidelines: Why will the algo work?

Let us not forget how hard it is to predict asset returns. Here we will try to gain an edge over the market by cleverly combining different signals coming from different approaches. These approaches will come from two main different types of predictive algorithm: Technical analysis and Machine Learning. It is very important to have a set of classic algo trading strats well calibrated to begin with. Indeed, the ML algos can only provide an edge if we use them with the right calibration approach, but it is a mistake to believe they can learn everything by themselves if they don’t have the right dataset, hence we need to be very careful about how we will treat the results of these algos. A clever approach for example would be to use data science method to tune hyperparameters of classic trading strats. 

Investigate the stock return patterns such as short term mean momentum, medium term mean reverting, and calendar anomalies… (paper from Agron)

## Data

Working on the dataset is very important, please do not forget that whichever algo we use, we cannot create new information, only try to describe it.

The data will be recovered from BBG or other source for the first daily dataset. Then we can use the API stream to build live dataset.

We can also use other data from the API, such as Quantopian and Quantdl datasets.

Here is the type of data we could include, in order of estimated importance:
* Price data (Close for daily, and Bid/Ask for Intraday)
* Traded Volume
* News Data (please see the part about news analysis for more details)
* Global Economic Data (CPI, GDP Growth, Interest Rates, ...)
* Single Stock Data (see paper)
* For other ideas, check the papers

Note that on the main reference and the second paper they transform an inomogeneous timestamp price dataset into a more simple time series containing only trend information, this approach could be very interesting but we would first need to check if this can be done IS consistently.

## Description of the strategy

Multi algo approach where a core algorithm uses a set of predictions given by a selection of predictive models to give a global answer.

Global hyperparameters:
*	n: numbers of total obs
*	h: time between two obs, crucial:
    * we will begin with one day and then test with intraday data
*	X: main dataset, stored once and then called by ref, see Data part
*	Y: price data, do we transform it?
*	Prediction type:
   * A binary output: Up/Dowm
   * A three clqss output: Up/Down/Neutral given a symetric threshold
   * A regression output

## Core algorithm

Ensemble method.
Basic prediction problem: predict a Y given a X.
We can give him more data than the outputs, especially data helping him to understand the different inputs he has.
Aim: Make sure the core algo always out perform all the single algos.

First approaches:
*	Pick best one out of a sample of obs
*	Equally weighted average
*	Manually fix the weights, allow for single algo testing

More advanced:
* Use any one of the algos as a the core algo
* Boosting algo methodology (for ex: AdaBoost):
    * on a linear reg w/out regularization
    * other base estimators
* Voting methodology :
    * Majority Vote 
    * Weighted Majority Vote
    * Borda Count 
    * Weighted Borda Count
    * Bayesian Formalism
    * BKS
    * Dempster–Shafer theories of evidence
* ESN: Echo State Network, see ref paper
* Markov Network

## Algos

All the algo needs to have a similar structure, hence we will build a general abstract class algo with the following attributes:
*	Predict function: function that gives a Y_test given a X_test
*	Select data function: manual or algorithmic way to select variables from the main dataset
*	Calib function: function that will train and calibrate the model on a Y_train and X_train subsets of obs. It will include the calibration of hyperparameters (check bellow)

The algos will be coded as subclass of the generic algo class, with overloaded methods. Here are a sample of possible algos:

*	Technical Analysis: cf ref for details
    * Historical Mean: arithmetic and geometric
    * Golden Cross / Dead Cross
    * MA enveloppe
    * RSI: Relative Strengh Index + Slopes
    * ROC: Rate of Change system
    * PSY: Psycological Stability
    * MOM: Momentum
    * VR: Volume Ratio
    * OBV: On Balance Volume
    * DIS: Disparity
    * STOD: Stochastic Indicator
    * CCI: Commodity Chanel Index
    * EVM: Ease of Movement
    * Force Index
    * Candle Chart Indicators:
        * Hammer and hanging man
        * Dark Cloud River
        * Piercing line
        * Engulginf pattern

* Machine Learning:
    *	Random Forest
    * SVM: Support Vector Machine
    * RVM: Relevance Vector Machine
    *	Neural Networks:
        *	MLP: Multi Layer Perceptrons, including Deep Learning methods
        *	RNN: Recurrent Neural Networks, LSM: Liquid State Machines, ESN: Echo State Networks
        *	GAN: Generative Adversarial Networks, can be used in RNN, including conditional GAN
        *	DBN: Deep Belief Networks
   *	Logit with/out regularization (LASSO, Ridge, Elastic Net)
   *	Linear reg with/out regularization
   *	Pairs Trading, or other correlation methods, can be coupled with cointegration testing modeling
   *	Web scrapping (on news websites like FXstreet or ForexFactory)
   *	Mean revert/trend follow on news impact
   *	Adaptive boosting, and other boosting algos, used on several base estimators
   *	K mean clustering
   *	KNN
   * Bayesian network
   * Kalman Filter

These algos will have to be independently calibrated using one of these methods:
* Time series expanding/rolling window cross validation:
    * Grid Search: brute forcing a set of hyperparams
    * Random Search: similar with a random subset of all combinaison
* GA: Genetic Algorithm

## Possible improvment: News Analysis

If implementing news analysis by itself as one of the algo is possible, news analysis can also be used in the trading strategy to avoid taking position close to big news event. As a result, including news into the model can have a range of impacts that could sensibly change the final performance of the stategy. Including a robust News Analysis tool is a key to generate consistent Alpha.

Please check these databases:
* Accern Alphaone News Sentiment
* Sentdex Sentiment Analysis
* EventVestor
* Raventpack

## Trading Strategy

Out of our predictions we built a trading strat based on:
* The final prediction
* The variance of the predictions
* Other indicators?

The basic idea is to go long/short when we predict a significant move with consistency across the models.

In case of a regression approach: to calculate the consistency we can assume a N(0,1) (or estimate via NP estimation a law) on Pred/std(Pred) and test his significativity at several thresholds. We could then invest only if the prediction is statistically different from zero.

In a case of classification approach: we can use the third prediction class Null to avoid too weak signals. This would be directed by an hyperparameter that can be estimated using an historical vol approach (GARCH?).

We could invest with a size inversely proportional to the variance, to define the exact optimal functional form of the size as a function of the prediction and its variance, we would need to solve an easy optimization problem on the PnL.

To conclude it would be interesing to code it using an set of input risk criterias, and let the algo optimiwe the trading strategy as a result.

## Trading

We can trade using these following ways:
* Quantopian: this option might be complicated given the size of the code and the external libraries used, however we mighe be able to import external files such as a trading log and use it in the Quantopian environment.
* Zipline: a python library developped by quantopian for algo trading allowing to backtest and run algorithm, it seems to include part of quantopian data and Quantdl data
* IG: online broker providing a trading and date stream API
* IB: InteractiveBrokers, similar to IG


In order to comunicate with the trading API we might need to code in another language such as C++, or use HTTP protocols.


## References & useful links

Please note than when only the name of the paper is given, you can find it on the IC Library Search tool.

* Main paper, but pretty basic, to read first, results are mainly cheated because of overfitting:
Intelligent stock trading system based on improved technical analysis and Echo State Network

* Similar article focused on ESN by same authors than first one:
http://web.inf.ufpr.br/menotti/ci171-2015-2-1/files/seminario-Dennis-artigo.pdf

* Details about how to transform price data into "high level price data":
Intelligent stock trading system by turning point confirming and probabilistic reasoning

* More complex, here they use GA as a core algo to mix differents MLs methods to predict:
An evolutionary approach to the combination of multiple classifiers to predict a stock price index

* Paper about GA used in Finance:
http://www.aiecon.org/staff/shc/pdf/IDEAL983.pdf

* About the Stochastic Acceptance Algorithm used in the Selection Phase of the GA:
https://arxiv.org/pdf/1109.3627.pdf

* Interesting paper, it covers the use of NN for time series prediction, seems fancy ass complicated tho:
https://arxiv.org/pdf/1703.01887.pdf

* Not related to finance but about GA + ESN:
Genetic algorithm optimized double-reservoir echo state network for multi-regime time series prediction

* Very good paper using esemble method on ML algorithm to built a trading strategy, it covers lots of ML algo apart from NN. Then it uses a multi model AdaBoost algo as a core algo. They also use a RELIEF algorithm to select features. However it works with 3 month prediction and with lots of economic data, it also gives a lot of analysis of economic effects to justify the fact that some algo will outperform others in certain prediods. Again careful with results, risk of overfitting here:
Ensemble Committees for Stock Return Classification and Prediction

* Thesis, might use GAN for time series predicting, hard to find anything else on GAN + time series:
Representation Learning on Time Series with Symbolic Approximation and Deep Learning

* What seems to be one of the main currently available paper on GAN, but not on time series:
Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks

* Presentation of GAN:
https://arxiv.org/pdf/1406.2661v1.pdf

* Presentation of ESN:
https://pdfs.semanticscholar.org/8922/17bb82c11e6e2263178ed20ac23db6279c7a.pdf

* General article about when the historical mean can be beaten as a predictor and why:
http://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf

* Veru interesting article about why there is natural biais in human decision making
http://isites.harvard.edu/fs/docs/icb.topic470237.files/articles%20spring%202008/Judgement%20under%20uncertainty%20readings/belief%20in%20the%20law%20of%20small%20numbers.pdf

* Quantopian article about Mean Reverting algorithms, focused on news impact, very interesting:
http://quantopian.us5.list-manage2.com/track/click?u=4c1d0cf4959586b47ef210e9c&id=42172f3a9e&e=135bb3f5c5

* Reference paper on Short Term Mean Reversion:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=436663

* About the Trading Strat:
https://www.quantopian.com/posts/mad-portfolio-an-alternative-to-markowitz

* Website presenting lots of Neural Nets:
http://www.asimovinstitute.org/neural-network-zoo/

* About Deep Learning in Finance, presentation given at IC:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2838013

* Python Library to easily use the IG API:
https://github.com/ig-python/ig-markets-api-python-library

* IG API:
http://labs.ig.com/apiorders

