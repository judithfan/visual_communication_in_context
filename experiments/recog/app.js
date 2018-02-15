global.__base = __dirname + '/';

var 
    use_https     = true,
    argv          = require('minimist')(process.argv.slice(2)),
    https         = require('https'),
    fs            = require('fs'),
    app           = require('express')(),
    _             = require('lodash'),
    parser        = require('xmldom').DOMParser,
    XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest,
    sendPostRequest = require('request').post;    

var gameport;

if(argv.gameport) {
  gameport = argv.gameport;
  console.log('using port ' + gameport);
} else {
  gameport = 8880;
  console.log('no gameport specified: using 8880\nUse the --gameport flag to change');
}

try {
  var privateKey  = fs.readFileSync('/etc/apache2/ssl/rxdhawkins.me.key'),
      certificate = fs.readFileSync('/etc/apache2/ssl/rxdhawkins.me.crt'),
      intermed    = fs.readFileSync('/etc/apache2/ssl/intermediate.crt'),
      options     = {key: privateKey, cert: certificate, ca: intermed},
      server      = require('https').createServer(options,app).listen(gameport),
      io          = require('socket.io')(server);
} catch (err) {
  console.log("cannot find SSL certificates; falling back to http");
  var server      = app.listen(gameport),
      io          = require('socket.io')(server);
}

app.get('/*', (req, res) => {
  serveFile(req, res); 
});

// var socket = io.connect('http://localhost:8001');

io.on('connection', function (socket) {
  socket.on('current_data', function(data) {
      console.log('current_data received: ' + JSON.stringify(data));
      writeDataToMongo(data);
  });

  socket.on('stroke', function(data) {
      console.log('stroke data received: ' + JSON.stringify(data));
      var xmlDoc = new parser().parseFromString(data['svg']);
      var svgData = xmlDoc.documentElement.getAttribute('d');
      data['svg'] = svgData;
      writeDataToMongo(data);      
  })

});

var UUID = function() {
  var baseName = (Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10));
  var template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
  var id = baseName + '-' + template.replace(/[xy]/g, function(c) {
    var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
  return id;
};

var serveFile = function(req, res) {
  var fileName = req.params[0];
  console.log('\t :: Express :: file requested: ' + fileName);
  return res.sendFile(fileName, {root: __dirname}); 
};

var writeDataToMongo = function(data) {
	  sendPostRequest(
    	'http://localhost:4000/db/insert',
    	{ json: data },
    	(error, res, body) => {
      if (!error && res.statusCode === 200) {
        console.log(`sent data to store`);
      } else {
		console.log(`error sending data to store: ${error} ${body}`);
      }
    }
  );
};

var gameID = UUID();

sendPostRequest('http://localhost:4000/db/getstims', {
  json: {dbname: 'stimuli', colname: 'sketchpad_basic_pilot2_sketches',
         numTrials: 10, gameid: gameID}
      }, (error, res, body) => {
        if (!error && res.statusCode === 200) {
          meta = body;
        } else {
            console.log(`error getting stims: ${error} ${body}`);
            console.log(`falling back to local stimList`);                     
        }
      }
})

// EDIT BELOW IN ORDER TO GET STIMS FROM DB

  // // if game relies on asynchronous stim logic, need to wait until everything
  // // is fetched before starting game (otherwise race conditions)
  // startGame(game) {
  //   if(game.experimentName == 'chairs_chatbox') {
  //     sendPostRequest('http://localhost:4000/db/getstims', {
  // json: {dbname: 'stimuli', colname: 'chairs1k',
  //        numRounds: game.numRounds, gameid: game.id}
  //     }, (error, res, body) => {
  // if(!error && res.statusCode === 200) {
  //         game.stimList = _.shuffle(body);
  //         game.trialList = game.makeTrialList();
  // } else {
  //   console.log(`error getting stims: ${error} ${body}`);
  //   console.log(`falling back to local stimList`);
  //   var closeFamilies = require('./stimList_chairs').closeByFamily;
  //   game.stimList = _.flatten(_.sampleSize(closeFamilies, game.numRounds));
  //   game.trialList = game.makeTrialList();
  // }
  // game.newRound();
  //     });
  //   } else {
  //     game.newRound();
  //   }
  // }

