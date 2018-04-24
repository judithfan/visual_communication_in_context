var _ = require('lodash');
var fs = require('fs');
var babyparse = require('babyparse');
// var JSONStream = require('JSONStream');
// var es = require('event-stream');

var getSimilarities = function(name) {
  return {
      //'mid-layer-triplet' : require('./json/similarity-splitbycontext-triplet_bugfix.json'),
      // 'mid-layer': require('./json/similarity-splitbycontext.json'),
      // 'early-layer': require('./json/similarity-splitbycontext-fixedpose_pool1.json'),
      //'mid-layer-augmented': require('./json/strict-similarity-pragmatics-fixedpose-augmented-splitbycontext_conv4_2.json'),
      'human': require('./json/similarity-human.json'),
      'fc6':  require('./json/similarity-fc6-centroid.json'),
      'sketch-unroll-synthetic': require('./json/similarity-splitbyobject-sketch_unroll_synthetic.json')
  };
};

var getCosts = function(name) {
  return require('./json/costs-' + name + '.json');
};

var getPossibleSketches = function(data) {
  return _.map(data, 'sketchLabel');
};

var getConditionLookup = function() {
  return require('../bdaInput/condition-lookup.json');
};

function _logsumexp(a) {
  var m = Math.max.apply(null, a);
  var sum = 0;
  for (var i = 0; i < a.length; ++i) {
    sum += (a[i] === -Infinity ? 0 : Math.exp(a[i] - m));
  }
  return m + Math.log(sum);
}

// P(target | sketch) = e^{scale * sim(t, s)} / (\sum_{i} e^{scale * sim(t, s)})
// => log(p) = scale * sim(target, sketch) - log(\sum_{i} e^{scale * sim(t, s)})
var getL0score = function(target, sketch, context, params, config) {
  var similarities = config.similarities[params.perception];
  var scores = [];
  for(var i=0; i<context.length; i++){
    var similarity = (similarities[context[i]][sketch]); // transform to range from 0 to 1
    scores.push(params.simScaling * similarity);
  }
  var similarity = (similarities[target][sketch]);
  return params.simScaling * similarity - _logsumexp(scores);
};

// Interpolates between the 'informativity' term of S0 and S1 based on pragWeight param
// Try remapping these to [0,1]...
var informativity = function(targetObj, sketch, context, params, config) {
  var sim = config.similarities[params.perception];
  var S0inf = (sim[targetObj][sketch]);// + 1.001) / 2;
//  console.log(S0inf);
  var S1inf = getL0score(targetObj, sketch, context, params, config); //Math.exp()
  // console.log(targetObj);
  // console.log(sketch);
  // console.log(S1inf);
  return ((1 - params.pragWeight) * S0inf + params.pragWeight * S1inf);
};

// note using logsumexp here isn't strictly necessary, because all the scores
// are *negative* (informativity is log(p of listener)) and there aren't
// enough terms for precision to matter...
var getSpeakerScore = function(trueSketch, targetObj, context, params, config) {
  var possibleSketches = config.possibleSketches;
  var costw = params.costWeight;
  var scores = [];
  // note: could memoize this for moderate optimization...
  // (only needs to be computed once per context per param, not for every sketch)
  for(var i=0; i<possibleSketches.length; i++){
    var sketch = possibleSketches[i];
    var inf = informativity(targetObj, sketch, context, params, config);
    var cost = config.costs[sketch];
    var utility = (1 - costw) * inf - costw * cost;
    scores.push(params.alpha * utility);//Math.log(Math.max(utility, Number.EPSILON)));
  }
  var trueUtility = ((1-costw) * informativity(targetObj, trueSketch, context, params, config)
		     - costw * config.costs[trueSketch]);
  //var roundedUtility = Math.max(trueUtility, Number.EPSILON);
  // console.log(_logsumexp(scores))
  //console.log(params.alpha * Math.log(roundedUtility))// - _logsumexp(scores));
  return params.alpha * trueUtility - _logsumexp(scores);
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

var paramSupportWriter = function(s, p, handle) {
  var sLst = _.toPairs(s);
  var l = sLst.length;

  for (var i = 0; i < l; i++) {
    fs.writeSync(handle, i + sLst[i].join(',')+','+p+'\n');
  }
};

var predictiveSupportWriter = function(s, filePrefix) {
  var predictiveFile = fs.openSync(filePrefix + "Predictives.csv", 'w');
  fs.writeSync(predictiveFile, ['index','game', "condition", 'trueSketch', "Target",
				"Distractor1", "Distractor2", "Distractor3",
				"coarseGrainedTrueSketch", "coarseGrainedPossibleSketch",
				"modelProb"] + '\n');

  var l = s.length;
  for (var i = 0; i < l; i++) {
    fs.writeSync(predictiveFile, s[i] + '\n');
  }

  fs.closeSync(predictiveFile);
};

// Note this is highly specific to a single type of erp
var bayesianErpWriter = function(erp, filePrefix) {
  var supp = erp.support();

  if(_.has(supp[0], 'params')) {
    var paramFile = fs.openSync(filePrefix + "Params.csv", 'w');
    fs.writeSync(paramFile, ["id", "perception", "pragmatics", "production", "alpha",
			     "simScaling", "pragWeight","costWeight",
			     "logLikelihood", "posteriorProb"] + '\n');
  }

  supp.forEach(function(s) {
    if(_.has(s, 'params'))
      paramSupportWriter(s.params, erp.score(s), paramFile);
  });
  if(_.has(supp[0], 'params')) {
    fs.closeSync(paramFile);
  }
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
  getSimilarities, getPossibleSketches, getCosts, getSubset, getConditionLookup,
  predictiveSupportWriter,
  getL0score, getSpeakerScore,
  bayesianErpWriter, writeCSV, readCSV, locParse
};
