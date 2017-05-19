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


      $word=strtolower($_GET["word"]);
      //echo '<script type="text/javascript" src="../outputs/'.$folder.'/dataset.json"></script>';
      $a=array();
      //$b="adele";
      $file="dataset_auto_mean.json";
      $json=json_decode(file_get_contents($file), true);
      if($word){
        for($i=0;$i<sizeof($json);$i++) if((strrpos(strtolower($json[$i]['artist']),$word)!==false)||(strrpos(strtolower($json[$i]['title']),$word)!==false)) array_push($a,$json[$i]);
        $json=$a;
      }
      shuffle($json);
      $json=array_slice($json,0,1000);

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
      var mix=-44;
      var mnx=52;
      var miy=-22;
      var mny=29;
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
      var previous=0;
      var now=0;
      var kind;
      var s;

      var songs=<?php echo json_encode($json, JSON_PRETTY_PRINT) ?>;
      var data=songs;

      function init(){
        if((!playlistmode)&&(!stop)){
          if(first){
            stage = new createjs.Stage("demoCanvas");
            stage.enableDOMEvents(true);
            z1=1;
            z2=1;
            stage.enableMouseOver(20);
            /*for(i=0;i<songs.length;i++){
              if(Math.abs(songs[i].x)>rangex) rangex=Math.ceil(Math.abs(songs[i].x));
              if(Math.abs(songs[i].y)>rangey) rangey=Math.ceil(Math.abs(songs[i].y));
            }*/
          }
          document.getElementById("drlist").style.color="#777777";
          document.getElementById("drlist").style.width="180px";
          document.getElementById("drlist").value="Create Playlist";
          document.getElementById("previous").style.visibility="hidden";
          document.getElementById("next").style.visibility="hidden";
          document.getElementById("previous").style.width="10px";
          document.getElementById("next").style.width="10px";
          stage.removeAllChildren();
          kind=document.getElementById("dgenre").value;
          var one=document.getElementById("coloruno").value.toLowerCase();
          var two=document.getElementById("colordue").value.toLowerCase();
          var three=document.getElementById("colortre").value.toLowerCase();
          n=1000;
          songs=data.slice(0,n);
          if(songs.length<200){
            mzoom=40;
          }
          for(i=0;i<songs.length;i++){
            var circle = new createjs.Shape();
            var bound = new createjs.Shape();
            var text = new createjs.Text("Artist: "+songs[i].artist+"\nTitle: "+songs[i].title, "40px Arial", "#777777");
            circle.x = ((songs[i].x+Math.abs(mix))*(demoCanvas.width)/(mnx-mix));
            circle.y = ((songs[i].y+Math.abs(miy))*(demoCanvas.height)/(mny-miy));
            bound.x= circle.x;
            bound.y= circle.y;
            text.x=circle.x+1;
            text.y=circle.y;
            text.textBaseline = "alphabetic";
            text.alpha=0;
            s=10;
            if(songs.length<200) s=2;
            if(stage.scaleX>s) text.alpha=1;
            if(stage.scaleX>2.10) bound.alpha=0;
            var c=Math.tanh(songs[i][one]+1);
            var m=Math.tanh(songs[i][two]+1);
            var y=Math.tanh(songs[i][three]+1);
            var k=0;//(songs[i].Angry)/7;
            var a=1;
            var r= Math.round(255*(1 - Math.min(1, c * (1 - k) + k)));
            var g= Math.round(255*(1 - Math.min(1, m * (1 - k) + k)));
            var b= Math.round(255*(1 - Math.min(1, y * (1 - k) + k)));
            //alert("r: "+r+" g: "+g+" b: "+b+" a: "+a);

            var e=Math.tanh(Math.pow(songs[i][kind]+1,3))*20+Math.pow((songs[i][kind]+1),3);
            circle.graphics.beginStroke("#777777").beginFill("rgba("+r+","+g+","+b+","+a+")").drawCircle(0, 0, 16+e);
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
            circle.addEventListener("click", function(event){
              if(previous.id!=event.target.id){
                playing(event.target);
              }
              for(i=0;i<playlistsong.length/3;i++){
                if(playlistsong[3*i+2]==event.target.id) now=i;
              }
            });
            circle.name="rgba("+r+","+g+","+b+",1)";
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
              document.getElementById("drlist").style.color="#777777";
              oldX=0;
              oldY=0;
              stop=true;
              playlistmode=false;
              for(i=songs.length;i<stage.numChildren;i=i+1){
                stage.removeChildAt(3*songs.length);
              }
              var note="Your playlist:\n\n";
              for(i=0;i<playlistsong.length/3;i++){
                var music=songs[playlistsong[3*i+2]];
                note=note+(i+1)+". "+music.artist+" - "+music.title+"\n";
              }
              alert(note);
              shape = new createjs.Shape();
              stage.addChild(shape);
              for(i=0;i<playlistsong.length-4;i=i+3){
                shape.graphics.beginStroke(colorw).setStrokeStyle(size, "round",0,10,true).moveTo(playlistsong[i],playlistsong[i+1]).lineTo(playlistsong[i+3],playlistsong[i+4]);
              }
              now=0;
              playing(stage.getChildAt(songs.length+2*playlistsong[2]));
              init();
            });
          });
        }
        if(first) playing(stage.getChildAt(3*songs.length-2));
        stage.update();
        first=false;
      }

      function playlist(){
        if(!playlistmode){
          playlistmode=true;
          document.getElementById("drlist").style.color="#fc3468";
          document.getElementById("drlist").style.width="90px";
          document.getElementById("drlist").value="Playlist";
          document.getElementById("previous").style.width="80px";
          document.getElementById("next").style.width="80px";
          document.getElementById("previous").style.visibility="visible";
          document.getElementById("next").style.visibility="visible";
          playlistsong=new Array();
          stage.removeAllEventListeners("stagemousedown");
          init();
        }
        else{
          playlistmode=false;
          stage.removeAllEventListeners("stagemousemove");
          stage.removeAllEventListeners("stagemousedown");
          line=false;
          document.getElementById("drlist").style.color="#777777";
          oldX=0;
          oldY=0;
          now=0;
          stop=true;
          for(i=songs.length;i<stage.numChildren;i=i+1){
            stage.removeChildAt(3*songs.length);
          }
          document.getElementById("drlist").style.color="#777777";
          document.getElementById("drlist").style.width="180px";
          document.getElementById("drlist").value="Create Playlist";
          document.getElementById("previous").style.visibility="hidden";
          document.getElementById("next").style.visibility="hidden";
          document.getElementById("previous").style.width="10px";
          document.getElementById("next").style.width="10px";
          playlistsong=new Array();
          init();
        }
      }

      function prec(){
        if(now){
          playing(stage.getChildAt(songs.length+2*playlistsong[3*(now-1)+2]));
          now--;
        }
      }

      function next(){
        if(now<(playlistsong.length/3-1)){
          playing(stage.getChildAt(songs.length+2*playlistsong[3*(now+1)+2]));
          now++;
        }
      }

      function playing(s){
        if(!first) document.getElementById("player").setAttribute("src",'https://embed.spotify.com/?uri=spotify%3Atrack%3A'+songs[s.id]["spotifyID"]+'&theme=white');
        e=Math.tanh(Math.pow(songs[s.id][kind]+1,3))*20+Math.pow((songs[s.id][kind]+1),3);
        s.graphics.clear().beginStroke(s.name).beginFill("#ffffff").drawCircle(0, 0, 16+e).endFill();
        s.scaleX*=1.8;
        s.scaleY*=1.8;
        if(previous){
          var color=stage.getChildAt(songs.length+2*previous.id).name;
          e=Math.tanh(Math.pow(songs[previous.id][kind]+1,3))*20+Math.pow((songs[previous.id][kind]+1),3);
          stage.getChildAt(songs.length+2*previous.id).graphics.beginStroke("#777777").beginFill(color).drawCircle(0,0,16+e).endFill();
          stage.getChildAt(songs.length+2*previous.id).scaleX/=1.8;
          stage.getChildAt(songs.length+2*previous.id).scaleY/=1.8;
          stage.update();
        }
        previous=s;
        stage.update();
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
              if(songs.length<200){
                z1=1/1.06;
                z2=1.06;
              }
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
              if(songs.length<200){
                z1=1.06;
                z2=1/1.06;
              }
              b=-0.08;
            }
          }
          var local = stage.globalToLocal(stage.mouseX, stage.mouseY);
          stage.regX=local.x;
          stage.regY=local.y;
          stage.x=stage.mouseX;
          stage.y=stage.mouseY;
          stage.scaleX=stage.scaleY*=z2;
          if(stage.scaleX>s) for(i=0;i<songs.length;i++){
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
              previous.scaleX=1.8;
              previous.scaleY=1.8;
            }
          }
          if((stage.scaleX>=mzoom)||(stage.scaleY>=mzoom)){
            stage.scaleX=mzoom;
            stage.scaleY=mzoom;
            resizing=true;
            for(i=0;i<3*songs.length;i++){
              stage.getChildAt(i).scaleX=1/mzoom;
              stage.getChildAt(i).scaleY=1/mzoom;
              previous.scaleX=1.8/mzoom;
              previous.scaleY=1.8/mzoom;
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
        <a class="navbar-brand" href="?word=">Music Visualisation with t-sne</a>
        <form action="<?php $_PHP_SELF ?>" method="GET">
            <input class="drop2" type="submit" value="Search"/>
            <input class="drop0" type="text" name="word" value=<?php echo "'".$_GET["word"]."'"?>/>
            <p style="float:right; margin-top:15px; margin-right:10px"> Type artist or word: </p>
        </form>
      </div>
    </nav>

    <!-- Begin page content -->
    <div class="container">
      <canvas id="demoCanvas" width="4000" height="1560"> </canvas>
    </div>

    <footer class="footer">
      <div class="explorer">
        <div class="vertical">
          <iframe id="player" src="https://embed.spotify.com/?uri=spotify%3Atrack%3A<?php echo $json[0]['spotifyID'] ?>&theme=white"frameborder="0" allowtransparency="true"></iframe>
        </div>
        <input id="drlist" class="drop" type="button" onclick="playlist()" value="Create playlist">
        <input id="previous" class="manager" style="visibility:hidden" type="button" onclick="prec()" value="Previous">
        <input id="next" class="manager" style="visibility:hidden" type="button" onclick="next()" value="Next">
        <div style="min-width:540px; overflow: hidden">
          <select class="drop1" id="coloruno" style="color: cyan; text-shadow: 1px 0 2px #777777;" onchange="init()">
            <option value"sad">Sad</option>
            <option value"angry">Angry</option>
            <option value"joy">Joy</option>
            <option value"erotic">Erotic</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <select class="drop1" id="colordue" style="color: magenta; text-shadow: 1px 0 2px #777777;" onchange="init()">
            <option value"erotic">Erotic</option>
            <option value"joy">Joy</option>
            <option value"angry">Angry</option>
            <option value"sad">Sad</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <select class="drop1" id="colortre" style="color: yellow; text-shadow: 1px 0 2px #777777;" onchange="init()">
            <option value"joy">Joy</option>
            <option value"sad">Sad</option>
            <option value"angry">Angry</option>
            <option value"erotic">Erotic</option>
            <option value"tender">Tender</option>
            <option value"fear">Fear</option>
          </select>
          <select id="dgenre" class="drop1" onchange="init()">
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
          <p style="float:right; margin-top:35px"> Feature size and color: </p>
      </div>
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
