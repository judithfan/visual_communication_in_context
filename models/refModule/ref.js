var _ = require('lodash');
var fs = require('fs');
var babyparse = require('babyparse');
// var JSONStream = require('JSONStream');
// var es = require('event-stream');

var getSimilarities = function(name) {
  return {
    'mid-layer' : require('./json/similarity-splitbycontext.json')
  };
};

var getCosts = function(name) {
  return require('./json/costs-' + name + '.json');
};

var getPossibleSketches = function(costs) {
  return _.keys(costs);
};

function _logsumexp(a) {
  var m = Math.max.apply(null, a);
  var sum = 0;
  for (var i = 0; i < a.length; ++i) {
    sum += (a[i] === -Infinity ? 0 : Math.exp(a[i] - m));
  }
  return m + Math.log(sum);
}

// P(target | sketch) \propto e^{scale * sim(t, s)}
// => log(p) = scale * sim(target, sketch) - log(\sum_{i} e^{scale * sim(t, s)})
var getL0score = function(target, sketch, context, params) {
  var similarities = params.similarities[params.similarityMetric];
  var scores = [];
  for(var i=0; i<context.length; i++){
    scores.push(params.simScale * similarities[context[i]][sketch]);
  }
  return params.simScale * similarities[target][sketch] - _logsumexp(scores);
};

// note using logsumexp here isn't strictly necessary, because all the scores
// are *negative* (informativity is log(p of listener)) and there aren't
// enough terms for precision to matter... 
var getCombinedScore = function(trueSketch, targetObj, context, params) {
  var possibleSketches = params.possibleSketches;
  var costw = params.costWeight;
  var pragw = params.costWeight;  
  var scores = [];
  for(var i=0; i<possibleSketches.length; i++){
    var sketch = possibleSketches[i];
    var inf = getL0score(targetObj, sketch, context, params);
    var cost = params.costs[sketch][0];
    var utility = (1 - w) * inf - w * cost;
    scores.push(params.alpha * utility);
  }
  var trueUtility = ((1 - w) * getL0score(targetObj, trueSketch, context, params)
		     - w * params.costs[trueSketch][0]);
  return params.alpha * trueUtility - _logsumexp(scores);
};

// note using logsumexp here isn't strictly necessary, because all the scores
// are *negative* (informativity is log(p of listener)) and there aren't
// enough terms for precision to matter... 
var getS1score = function(trueSketch, targetObj, context, params) {
  var possibleSketches = params.possibleSketches;
  var w = params.costWeight;
  var scores = [];
  for(var i=0; i<possibleSketches.length; i++){
    var sketch = possibleSketches[i];
    var inf = getL0score(targetObj, sketch, context, params);
    var cost = params.costs[sketch][0];
    var utility = (1 - w) * inf - w * cost;
    scores.push(params.alpha * utility);
  }
  var trueUtility = ((1 - w) * getL0score(targetObj, trueSketch, context, params)
		     - w * params.costs[trueSketch][0]);
  return params.alpha * trueUtility - _logsumexp(scores);
};

var getS0score = function(trueSketch, targetObj, params) {
  var possibleSketches = params.possibleSketches;
  var similarities = params.similarities[params.similarityMetric];
  var w = params.costWeight;
  var scores = [];
  for(var i=0; i<possibleSketches.length; i++){
    var sketch = possibleSketches[i];
    var inf = similarities[targetObj][sketch];
    var cost = params.costs[sketch][0];
    var utility = (1 - w) * inf - w * cost;
    scores.push(params.alpha * utility);
  }
  var trueUtility = ((1-w) * similarities[targetObj][trueSketch]
		     - w * params.costs[trueSketch][0]);
  
  return Math.log(Math.exp(params.alpha * trueUtility)) - _logsumexp(scores);
};

function readCSV(filename){
  return babyparse.parse(fs.readFileSync(filename, 'utf8'),
			 {header:true, skipEmptyLines:true}).data;
};

function writeCSV(jsonCSV, filename){
  fs.writeFileSync(filename, babyparse.unparse(jsonCSV) + '\n');
}

function appendCSV(jsonCSV, filename){
  fs.appendFileSync(filename, babyparse.unparse(jsonCSV) + '\n');
}

var supportWriter = function(s, p, handle) {
  var sLst = _.toPairs(s);
  var l = sLst.length;

  for (var i = 0; i < l; i++) {
    fs.writeSync(handle, sLst[i].join(',')+','+p+'\n');
  }
};

// Note this is highly specific to a single type of erp
var bayesianErpWriter = function(erp, filePrefix) {
  
  var predictiveFile = fs.openSync(filePrefix + "Predictives.csv", 'w');
  fs.writeSync(predictiveFile, ["condition", "Target", "Distractor1", "Distractor2", "Distractor3", 
				"value", "prob", "posteriorProb"] + '\n');

  var paramFile = fs.openSync(filePrefix + "Params.csv", 'w');
  fs.writeSync(paramFile, ["similarityMetric,", "speakerModel", "alpha", "typWeight", "costWeight", "logLikelihood", "posteriorProb"] + '\n');

  var supp = erp.support();
 
  supp.forEach(function(s) {
    supportWriter(s.predictive, erp.score(s), predictiveFile);
    supportWriter(s.params, erp.score(s), paramFile);
  });
  fs.closeSync(predictiveFile);
  fs.closeSync(paramFile);
  console.log('writing complete.');
};

var getSubset = function(data, properties) {
  var matchProps = _.matches(properties);
  return _.filter(data, matchProps);
};

var locParse = function(filename) {
  return babyparse.parse(fs.readFileSync(filename, 'utf8'),
       {header: true,
        skipEmptyLines : true}).data;
};

module.exports = {
  getSimilarities, getPossibleSketches, getCosts, getSubset,
  getL0score, getS1score, getS0score,
  bayesianErpWriter, writeCSV, readCSV, locParse
};
