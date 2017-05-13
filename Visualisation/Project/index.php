<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="icon/play.png">

    <title>Visualisation</title>

    <!-- Bootstrap core CSS -->
    <link href="bootstrap-3.3.7/docs/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <link href="bootstrap-3.3.7/docs/assets/css/ie10-viewport-bug-workaround.css" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="sticky-footer-navbar.css" rel="stylesheet">

    <script src="https://code.createjs.com/easeljs-0.8.2.min.js"></script>

    <script src="https://code.createjs.com/tweenjs-0.6.2.min.js"></script>

    <script src="http://d3js.org/d3.v3.min.js" charset="utf-8"></script>

    <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"> </script>

    <?php


      if(empty($_GET["folder"])) $folder="20170502_104427_5000kullback_tsneLoss-batch80-epochs150";
      else $folder=$_GET["folder"];
      echo '<script type="text/javascript" src="../outputs/'.$folder.'/dataset.json"></script>';
      $file="../outputs/".$folder."/dataset.json";

    ?>

    <!--script>
      // Create a variable with the url to the JSON file
      var url = "https://gist.githubusercontent.com/d3byex/e5ce6526ba2208014379/raw/8fefb14cc18f0440dc00248f23cbf6aec80dcc13/walking_dead_s5.json";

      // Load the json file
      d3.json(url, function(error, data) {
          // Output the first observation to the log
          console.log(data[0]);
      });
    </script-->

    <script>

      var isSafari = /Safari/.test(navigator.userAgent) && /Apple Computer/.test(navigator.vendor);

      var first=true;
      var stage;
      var size=6;
      var colorw="#a7adba";
      var mzoom=100;
      var rangex=0;
      var rangey=0;
      var z1;
      var z2;
      var one;
      var two;
      var three;
      var oldX;
      var oldY;
      var shape;
      var playlistmode=false;
      var line=false;
      var playlistsong;
      var stop=false;


      <?php
        $json=json_decode(file_get_contents($file), true);
      ?>

      var songs=<?php echo json_encode($json, JSON_PRETTY_PRINT) ?>;;
      var data=songs;

      function init(){
        if((!playlistmode)&&(!stop)){
          if(first){
            stage = new createjs.Stage("demoCanvas");
            stage.enableDOMEvents(true);
            z1=1;
            z2=1;
            stage.enableMouseOver(20);
            for(i=0;i<songs.length;i++){
              if(Math.abs(songs[i].x)>rangex) rangex=Math.ceil(Math.abs(songs[i].x));
              if(Math.abs(songs[i].y)>rangey) rangey=Math.ceil(Math.abs(songs[i].y));
            }
          }
          stage.removeAllChildren();
          var kind=document.getElementsByClassName("drop")[0].value;
          var one=document.getElementById("coloruno").value.toLowerCase();
          var two=document.getElementById("colordue").value.toLowerCase();
          var three=document.getElementById("colortre").value.toLowerCase();
          n=document.getElementsByName("ns")[0].value;
          songs=data.slice(0,n);
          for(i=0;i<songs.length;i++){
            var circle = new createjs.Shape();
            var bound = new createjs.Shape();
            var text = new createjs.Text("Artist: "+songs[i].artist+"\nTitle: "+songs[i].title, "40px Arial", "#777777");
            circle.x = ((songs[i].x+rangex)*(demoCanvas.width)/(2*rangex));
            circle.y = ((songs[i].y+rangey)*(demoCanvas.height-100)/(2*rangey));
            bound.x= circle.x;
            bound.y= circle.y;
            text.x=circle.x+1;
            text.y=circle.y;
            text.textBaseline = "alphabetic";
            text.alpha=0;
            if(stage.scaleX>0.1*mzoom) text.alpha=1;
            if(stage.scaleX>2.10) bound.alpha=0;
            var c=(songs[i][one]+1)/2;
            var m=(songs[i][two]+1)/2;
            var y=(songs[i][three]+1)/2;
            var k=0;//(songs[i].Angry)/7;
            var a=1;
            var r= Math.round(255*(1 - Math.min(1, c * (1 - k) + k)));
            var g= Math.round(255*(1 - Math.min(1, m * (1 - k) + k)));
            var b= Math.round(255*(1 - Math.min(1, y * (1 - k) + k)));
            //alert("r: "+r+" g: "+g+" b: "+b+" a: "+a);

            var e=Math.pow(Math.round(songs[i][kind]+1)*2,2);
            circle.graphics.beginStroke("#777777").beginFill("rgba("+r+","+g+","+b+","+a+")").drawCircle(0, 0, 12+e);
            bound.graphics.beginRadialGradientFill(["rgba("+r+","+g+","+b+",0.1)","rgba("+r+","+g+","+b+",0)"],[0, 1],0,0,10,0,0,100).drawCircle(0,0,100);
            text.shadow = new createjs.Shadow("rgba("+r+","+g+","+b+",0.2)", 5, 5, 10);
            circle.id=i;
            circle.addEventListener("mouseover", function(event) {
              if((line)&&(playlistsong[playlistsong.length-1]!=event.target.id)){
                playlistsong.push(event.target.x);
                playlistsong.push(event.target.y);
                playlistsong.push(event.target.id);
              }
              event.target.scaleX*=1.8;
              event.target.scaleY*=1.8;
              stage.update();
            });
            circle.addEventListener("mouseout", function(event) {
              if((line)&&(playlistsong[playlistsong.length-1]!=event.target.id)){
                playlistsong.push(event.target.x);
                playlistsong.push(event.target.y);
                playlistsong.push(event.target.id);
              }
              event.target.scaleX/=1.8;
              event.target.scaleY/=1.8;
              stage.update();
            });
            circle.addEventListener("click", function(event) {
              var title=songs[event.target.id].title;
              var artist=songs[event.target.id].artist;
              alert("title: "+title+"\nartist: "+artist);
              //document.getElementById("player").setAttribute("src",'https://embed.spotify.com/?uri=spotify%3Atrack%3A'+song+'&theme=white');
              //document.getElementsByClassName("footer")[0].style.backgroundColor="#f4d5fa";
            });
            circle.scaleX/=stage.scaleX;
            circle.scaleY/=stage.scaleY;
            bound.scaleX/=stage.scaleX;
            bound.scaleY/=stage.scaleY;
            text.scaleX/=stage.scaleX;
            text.scaleY/=stage.scaleY;

            stage.addChild(circle);
            stage.addChild(bound);
            stage.addChild(text);
            stage.setChildIndex(bound,0);
          }
        }
        if(first||stop){
          demoCanvas.addEventListener("mousewheel", MouseWheelHandler, false);
          demoCanvas.addEventListener("DOMMouseScroll", MouseWheelHandler, false);
          if(stop){
            stage.removeAllEventListeners("stagemousedown");
            stage.removeAllEventListeners("stagemouseup");
            stage.removeAllEventListeners("stagemousemove");
          }
          stage.addEventListener("stagemousedown", function(e) {
            var offset={x:stage.x-e.stageX,y:stage.y-e.stageY};
            stage.addEventListener("stagemousemove",function(ev) {
            	stage.x = ev.stageX+offset.x;
            	stage.y = ev.stageY+offset.y;
            	stage.update();
            });
            stage.addEventListener("stagemouseup", function(){
            	stage.removeAllEventListeners("stagemousemove");
            });
          });
          stop=false;
        }
        if(playlistmode){
          shape = new createjs.Shape();
          for(i=songs.length;i<stage.numChildren;i=i+1){
            stage.removeChildAt(3*songs.length);
          }
          stage.addChild(shape);
          stage.removeAllEventListeners("stagemousedown");
          stage.removeAllEventListeners("stagemouseup");
          stage.removeAllEventListeners("stagemousemove");
          stage.addEventListener("stagemousedown", function(e) {
            line=true;
            stage.on("stagemousemove",function(evt){
        		  if(oldX){
                var local=stage.globalToLocal(oldX,oldY);
                oldX=local.x;
                oldY=local.y;
                local=stage.globalToLocal(evt.stageX,evt.stageY);
        			  shape.graphics.beginStroke(colorw).setStrokeStyle(size/stage.scaleX, "round").moveTo(oldX, oldY).lineTo(local.x, local.y);
        			  stage.update();
        		  }
        		  oldX=evt.stageX;
        		  oldY=evt.stageY;
      		  });
            stage.addEventListener("stagemouseup", function(){
              stage.removeAllEventListeners("stagemousemove");
              stage.removeAllEventListeners("stagemousedown");
              line=false;
              oldX=0;
              oldY=0;
              stop=true;
              playlistmode=false;
              for(i=songs.length;i<stage.numChildren;i=i+1){
                stage.removeChildAt(3*songs.length);
              }
              //alert(playlistsong);
              shape = new createjs.Shape();
              stage.addChild(shape);
              for(i=0;i<playlistsong.length-4;i=i+3){
                shape.graphics.beginStroke(colorw).setStrokeStyle(size, "round",0,10,true).moveTo(playlistsong[i],playlistsong[i+1]).lineTo(playlistsong[i+3],playlistsong[i+4]);
              }
              stage.update();
              init();
            });
          });
        }
        stage.update();
        first=false;
      }

      function playlist(){
        playlistmode=true;
        playlistsong=new Array();
        stage.removeAllEventListeners("stagemousedown");
        init();
      }

      function MouseWheelHandler(e) {
          var resizing=false;
          var b;
        	if(Math.max(-1, Math.min(1, (e.wheelDelta || -e.detail)))>0)
          {
            if(isSafari){
              z1=5/mzoom;
              z2=mzoom/5;
            }
            else{
              //the larger the value of z2 the faster the speed
          		z1=1/1.11;
              z2=1.11;
              b=0.08;
            }
          }
        	else
          {
            if(isSafari){
              z1=mzoom/5;
              z2=5/mzoom;
            }
            else{
          		z1=1.11;
              z2=1/1.11;
              b=-0.08;
            }
          }
          var local = stage.globalToLocal(stage.mouseX, stage.mouseY);
          stage.regX=local.x;
          stage.regY=local.y;
          stage.x=stage.mouseX;
          stage.y=stage.mouseY;
          stage.scaleX=stage.scaleY*=z2;
          if(stage.scaleX>0.1*mzoom) for(i=0;i<songs.length;i++){
            stage.getChildAt(2*i+songs.length+1).alpha+=b;
            if(stage.getChildAt(2*i+songs.length+1).alpha>1) stage.getChildAt(2*i+songs.length+1).alpha=1;
            if(stage.getChildAt(2*i+songs.length+1).alpha<0) stage.getChildAt(2*i+songs.length+1).alpha=0;
          }
          else for(i=0;i<songs.length;i++){
            stage.getChildAt(2*i+songs.length+1).alpha=0;
          }
          if((stage.scaleX<=1)||(stage.scaleY<=1)){
            stage.scaleX=1;
            stage.scaleY=1;
            resizing=true;
            for(i=0;i<stage.numChildren-1;i++){
              stage.getChildAt(i).scaleX=1;
              stage.getChildAt(i).scaleY=1;
            }
          }
          if((stage.scaleX>=mzoom)||(stage.scaleY>=mzoom)){
            stage.scaleX=mzoom;
            stage.scaleY=mzoom;
            resizing=true;
            for(i=0;i<3*songs.length;i++){
              stage.getChildAt(i).scaleX=1/mzoom;
              stage.getChildAt(i).scaleY=1/mzoom;
            }
          }
          if(!resizing) for(i=0;i<3*songs.length;i++){
            stage.getChildAt(i).scaleX*=z1;
            stage.getChildAt(i).scaleY*=z1;
          }
          for(i=0;i<songs.length;i++){
            stage.getChildAt(i).alpha-=b;
            if(stage.getChildAt(i).alpha>1) stage.getChildAt(i).alpha=1;
            if(stage.getChildAt(i).alpha<0) stage.getChildAt(i).alpha=0;
          }
          /*
          mx=stage.mouseX*z2+2000;
          mix=stage.mouseX*z2-2000;
          my=stage.mouseY*z2+875;
          miy=stage.mouseX*z2-875;
          */
          stage.update();
        }
        
    </script>

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->
  </head>

  <body onload="init()">

    <!-- Fixed navbar -->
    <nav class="navbar navbar-default navbar-fixed-top" background-color=#b3cde0>
      <div class="container">
        <div class="navbar-header">
          <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
            <span class="sr-only">Toggle navigation</span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
          </button>
          <a class="navbar-brand" href="#">Music Visualisation with t-sne</a>
        </div>
          <select class="drop2" id="coloruno" style="color: cyan" onchange="init()">
            <option value"sad">Sad</option>
            <option value"angry">Angry</option>
            <option value"joy">Joy</option>
            <option value"erotic">Erotic</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <select class="drop2" id="colordue" style="color: magenta" onchange="init()">
            <option value"erotic">Erotic</option>
            <option value"joy">Joy</option>
            <option value"angry">Angry</option>
            <option value"sad">Sad</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <select class="drop2" id="colortre" style="color: yellow" onchange="init()">
            <option value"joy">Joy</option>
            <option value"sad">Sad</option>
            <option value"angry">Angry</option>
            <option value"erotic">Erotic</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <form action="<?php $_PHP_SELF ?>" method="GET">
            <input class="drop2" type="submit" value="Draw file"/>
            <input class="drop2" type="text" name="folder" value=<?php echo "'".$_GET["folder"]."'"?>/>
            <p style="float:right; margin-top:15px; margin-right:10px"> Choose folder: </p>
          </form>
        </div>
      </div>
    </nav>


    <!-- Begin page content -->
    <div class="container" margin-top:60px>
      <canvas id="demoCanvas" width="4000" height="1750"> </canvas>
    </div>

    <footer class="footer">
      <select class="drop" onchange="init()">
        <option value="blues">Blues</option>
        <option value="country">Country</option>
        <option value="easylistening">Easy Listening</option>
        <option value="electronica">Electronica</option>
        <option value="folk">Folk</option>
        <option value="hiphopurban">Hip Hop Urban</option>
        <option value="jazz">Jazz</option>
        <option value="latin">Latin</option>
        <option value="newage">New Age</option>
        <option value="pop">Pop</option>
        <option value="rnbsoul">R'n'b Soul</option>
        <option value="rock">Rock</option>
        <option value="gospel">Gospel</option>
        <option value="reggae">Reggae</option>
        <option value="world">World</option>
      </select>
      <div class="vertical">
        <iframe id="player" src="https://embed.spotify.com/?uri=spotify%3Atrack%3A6eZpo3Bp44hOkG2d1v8s5L&theme=white"frameborder="0" allowtransparency="true"></iframe>
      </div>
      <input class="drop1" type="button" onclick="playlist()" value="Draw">
      <input class="drop1" type="text" name="ns" value="1000">
    </footer>


    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
    <script>window.jQuery || document.write('<script src="bootstrap-3.3.7/docs/assets/js/vendor/jquery.min.js"><\/script>')</script>
    <script src="bootstrap-3.3.7/docs/dist/js/bootstrap.min.js"></script>
    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <script src="bootstrap-3.3.7/docs/assets/js/ie10-viewport-bug-workaround.js"></script>
  </body>
</html>
