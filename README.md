# MSTA
Multi Strategy Trading Algorithm

## Main guidelines: Why will the algo work?

Let us not forget how hard it is to predict stock returns. However, this is due to the very basic approach usually taken by people trying to predict. Here we will try to gain an edge over the market by cleverly combining different signals coming from different approaches. It is very important to have a set of classic algo trading strats well calibrated to begin with. Indeed, the ML algos can only provide an edge if we use them with the right calibration approach, but it is a mistake to believe they can learn everything by themselves if they don’t have the right dataset, hence we need to be very careful about how we will treat the results of these algos. A clever approach for example would be to use data science method to tune hyperparameters of classic trading strats. 
Investigate the stock return patterns such as short term mean momentum, medium term mean reverting, and calendar anomalies… (paper from Agron)

## Data

The data will be recovered from BBG or other source for the first daily dataset. Then we can use IG HTTP protocols to build dataset live and trade live.

Note that on the main reference paper they transform an inomogenous timestamp price dataset into a more simple time series containing only trend information.

## Description of the strategy

Multi algo approach where a core algorithm uses a set of predictions given by a selection of predictive models to give a global answer. As of now the prediction model will be restricted to a binary (up/down) classification equivalent to predicting if the return of an asset will be positive, allowing the use of classification algos.

Global hyperparameters:
*	n: numbers of total obs
*	h: time between two obs, crucial
*	X: main dataset, stored once and then called by ref, by default just historical price data
*	Y: what do we try to predict? Prices? Returns? Log returns?
*	Do we choose a binary prediction approach (up/down), or three classes (up/down/null) or a regression?

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
*	Calibrate the weights using one of the algos
*	Boosting algo methodology on a linear reg w/out regularization
*	Voting methodology (majority or more complexe)
* ESN

## Algos

All the algo needs to have a similar structure, hence we will build a general abstract class algo with the following attributes:
*	Predict function: function that gives a Y given a X
*	Select data function: manual or algo way to select variables from the main dataset
*	Train function: function that will train and calibrate the model on a Y and X subsets of obs

They will all be calibrated and called using the same syntax and then the same arguments.
The algos will be coded as subclass of the generic algo class, with overloaded methods. Here are a sample of possible algos:
*	Random Forest
*	SVM
*	Neural Networks:
    *	MLP: Multi Layer Perceptrons, including Deep Learning methods
    *	RNN: Recurrent Neural Networks, LSM: Liquid State Machines, ESN: Echo State Networks (good for TS)
    *	GAN: Generative Adversarial Networks, can be used in RNN, including conditional GAN
    *	DBN: Deep Belief Networks
*	Logit with/out LASSO, Ridge, Elastic Net
*	Linear reg with/out LASSO, Ridge, Elastic Net
*	Pair Trading, or other correlation methods, can be coupled with correlation modeling using ARIMA
*	Technical Analysis: cf ref for details
    * Golden Cross / Dead Cross
    * MA enveloppe
    * RSI: Relative Strengh Index + Slopes
    * ROC: Rate of Change system
    * Stochastic Indicator
    * Candle Chart Indicators:
        * Hammer and hanging man
        * Dark Cloud River
        * Piercing line
        * Engulginf pattern
*	Web scrapping (on news websites like FXstreet or ForexFactory)
*	Mean revert/trend follow on news impact
*	Adaptive boosting, and other boosting algos, used on several base estimators
*	K mean clustering
*	KNN
*	Bayesian network

These algos will have to be independently calibrated using one of these methods:
* Time series expanding/rolling window cross validation:
    * Grid Search: brute forcing a set of hyperparams
    * Random Search: similar with a random subset of all combinaison
* GA: Genetic Algorithm

## Trading Strategy

Out of our predictions we built a trading strat based on:
*	The final prediction
* The variance of the predictions
*	Other indicators?

The basic approach is to go long/short when we predict a significant move with consistency across the models. To calculate the consistence we can assume a N(0,1) (or estimate via NP estimation a law) on Pred/std(Pred) and test his significativity at several threshold. We could tehn invest only if the prediction is statistically different from zero. 
We could invest with a size inversely proportional to the variance, to define the exact optimal functional form of the size as a function of the prediction and its variance, we would need to solve an easy optimization problem on the PnL.

To trade we can either connect to IG using a python library or directly use Quantopian. The IG API also provides live price information!

## References & useful links

Intelligent stock trading system based on improved technical analysis and Echo State Network

https://github.com/ig-python/ig-markets-api-python-library

http://labs.ig.com/apiorders

