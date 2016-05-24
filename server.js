var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);
var config = require('./config.json');
var fb = require('./fogbugz.js')(config.fogbugz);

app.use(express.static('public'));

app.get('/show/:case',function(req,res) {
    var casenr = req.params.case;
    console.log('show',casenr);
    fb.getCase(casenr,['ixBug','sTitle','sProject','sPersonAssignedTo']).then(function(res) {
        console.log(res);
        io.sockets.emit('ball',{nr: casenr});
        res.send('ok');
    },function(err) {
        console.log(err)
    });
});

app.get('/add/:case',function(req,res) {
    var casenr = req.params.case;
    console.log('add',casenr);
    fb.getCase(casenr,['ixBug','sTitle','sProject','sPersonAssignedTo']).then(function(result) {
        console.log(result);
        io.sockets.emit('ball_add',result[0]);
        res.send('ok');
    },function(err) {
        console.log(err)
    });
});

app.get('/rem/:case',function(req,res) {
    var casenr = req.params.case;
    console.log('show',casenr);
    fb.getCase(casenr,['ixBug','sTitle','sProject','sPersonAssignedTo']).then(function(result) {
        console.log(result);
        io.sockets.emit('ball_remove',result[0]);
        res.send('ok');
    },function(err) {
        console.log(err)
    });
});

var port = 8477;
server.listen(port);
console.log('case server running on port',port)