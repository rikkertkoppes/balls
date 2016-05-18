var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);

app.use(express.static('public'));

app.get('/show/:case',function(req,res) {
    var casenr = req.params.case;
    console.log('show',casenr);
    io.sockets.emit('ball',{nr: casenr});
    res.send('ok');
});

var port = 8477;
server.listen(port);
console.log('case server running on port',port)