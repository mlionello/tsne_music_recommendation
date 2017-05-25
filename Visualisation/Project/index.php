<?php
// Start the session
session_start();
?>

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

    <link href="jquery-ui-1.12.1.custom/jquery-ui.css" rel="stylesheet">

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
      $fillings=array();
      //$b="adele";
      $file="dataset_auto_mean.json";
      $json=json_decode(file_get_contents($file), true);
      if(!isset($_SESSION['fillings'])){
        for($i=0;$i<sizeof($json);$i++){
          if(strlen($json[$i]['artist'])>15) $artist=substr($json[$i]['artist'],0,strpos($json[$i]['artist'],' ',15));
          else $artist=$json[$i]['artist'];
          if(strlen($artist)>20) $artist=substr($artist,0,20);
          if(strpos($artist,'\\')!==false||strpos($artist,'/')!==false) $artist=substr($artist,0,strpos($artist,'\\'));
          $keywords = preg_split("/[\s,]+/", $artist);
          for($j=0;$j<sizeof($keywords);$j++){
            if(!in_array($keywords[$j], $fillings, true)){
                array_push($fillings, $keywords[$j]);
            }
          }
          if(!in_array($artist, $fillings, true)) array_push($fillings, $artist);
          /*if(strlen($json[$i]['title'])>15) $title=substr($json[$i]['title'],0,strpos($json[$i]['title'],' ',15));
          else $title=$json[$i]['title'];
          if(strlen($title)>20) $title=substr($title,0,20);
          if(strpos($title,'\\')!==false) $title=substr($title,0,strpos($title,'\\'));
          if(!in_array($title, $fillings, true)) array_push($fillings, $title);*/
        }
        asort($fillings);
        $fillings=array_values($fillings);
        $_SESSION['fillings']=$fillings;
      }
      else $fillings=$_SESSION['fillings'];

      if($word){
        for($i=0;$i<sizeof($json);$i++) if((strrpos(strtolower($json[$i]['artist']),$word)!==false)||(strrpos(strtolower($json[$i]['title']),$word)!==false)) array_push($a,$json[$i]);
        $json=$a;
      }
      shuffle($json);
      $json=array_slice($json,0,1000);

      if(!isset($_GET["coltwo"])){
        $_GET["genres"]="blues";
        $_GET["colone"]="rock";
        $_GET["coltwo"]="erotic";
        $_GET["colthree"]="tender";
      }

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
      var colorw="rgba(167,173,186,0.9)";
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
      var playlistsong=new Array();
      var stop=false;
      var previous=0;
      var now=-1;
      var kind;
      var note="Your playlist:<br>";
      var s;

      var songs=<?php echo json_encode($json, JSON_PRETTY_PRINT) ?>;
      var data=songs;
      var setting=3*songs.length-2;

      function init(){
        if((!playlistmode)&&(!stop)){
          if(first){
            stage = new createjs.Stage("demoCanvas");
            stage.enableDOMEvents(true);
            z1=1;
            z2=1;
            stage.enableMouseOver(20);
            <?php
            if(!isset($_SESSION["load"])) echo "colorpopup();\nsetTimeout(colorpopup, 2000);";
            $_SESSION["load"]=true;
            ?>
            /*for(i=0;i<songs.length;i++){
              if(Math.abs(songs[i].x)>rangex) rangex=Math.ceil(Math.abs(songs[i].x));
              if(Math.abs(songs[i].y)>rangey) rangey=Math.ceil(Math.abs(songs[i].y));
            }*/
          }
          stage.removeAllChildren();
          kind=document.getElementById("dgenre").value.toLowerCase();
          var one=document.getElementById("coloruno").value.toLowerCase();
          var two=document.getElementById("colordue").value.toLowerCase();
          var three=document.getElementById("colortre").value.toLowerCase();
          document.getElementById("genres").setAttribute("value",kind);
          document.getElementById("colone").setAttribute("value",one);
          document.getElementById("coltwo").setAttribute("value",two);
          document.getElementById("colthree").setAttribute("value",three);
          document.getElementById("badge").setAttribute("href","?genres="+kind+"&colone="+one+"&coltwo="+two+"&colthree="+three);

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
            var k=Math.tanh((songs[i][three]+1)/18);//(songs[i].Angry)/7;
            var k=0;
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
              now=-1;
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
          if(!first) playing(stage.getChildAt(songs.length+2*previous.id));
          if(playlistsong.length>0) drawPlaylist();
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
              document.getElementById("drlist").setAttribute("src","icon/draw.png");
              oldX=0;
              oldY=0;
              stop=true;
              playlistmode=false;
              for(i=0;i<playlistsong.length/3;i++){
                var music=songs[playlistsong[3*i+2]];
                note=note+"<br>"+(i+1)+". "+music.artist+" - "+music.title;
              }
              overlay();
              drawPlaylist();
              init();
            });
          });
        }
        if(first) playing(stage.getChildAt(setting));
        stage.update();
        first=false;
      }

      function playlist(){
        if(!playlistmode){
          playlistmode=true;
          document.getElementById("drlist").setAttribute("src","icon/drawhigh.png");
          now=0;
          playlistsong=new Array();
          stage.removeAllEventListeners("stagemousedown");
          init();
        }
        else{
          playlistmode=false;
          stage.removeAllEventListeners("stagemousemove");
          stage.removeAllEventListeners("stagemousedown");
          line=false;
          document.getElementById("drlist").setAttribute("src","icon/draw.png");
          oldX=0;
          oldY=0;
          now=-1;
          stop=true;
          for(i=songs.length;i<stage.numChildren;i=i+1){
            stage.removeChildAt(3*songs.length);
          }
          document.getElementById("drlist").style.color="#777777";
          playlistsong=new Array();
          init();
        }
      }

      function drawPlaylist(){
        for(i=songs.length;i<stage.numChildren;i=i+1){
          stage.removeChildAt(3*songs.length);
        }
        shape = new createjs.Shape();
        stage.addChild(shape);
        for(i=0;i<playlistsong.length-4;i=i+3){
          shape.graphics.beginStroke(colorw).setStrokeStyle(size, "round",0,10,true).setStrokeDash([30, 10], 0).moveTo(playlistsong[i],playlistsong[i+1]).lineTo(playlistsong[i+3],playlistsong[i+4]);
        }
        if(previous.id!=stage.getChildAt(songs.length+2*playlistsong[3*now+2]).id){
          playing(stage.getChildAt(songs.length+2*playlistsong[3*now+2]));
        }
        stage.update();
      }

      function empty(){
        playlistmode=true;
        playlist();
      }

      function prec(){
        if(now>0){
          playing(stage.getChildAt(songs.length+2*playlistsong[3*(now-1)+2]));
          now--;
        }
        else if(now<0){
          setting=setting+2;
          if(setting>3*songs.length-2) setting=songs.length;
          playing(stage.getChildAt(setting));
        }
      }

      function next(){
        if((now>=0)&&(now<(playlistsong.length/3-1))){
          playing(stage.getChildAt(songs.length+2*playlistsong[3*(now+1)+2]));
          now++;
        }
        else if(now<0){
          setting=setting-2;
          if(setting<songs.length) setting=3*songs.length-2;
          playing(stage.getChildAt(setting));
        }
      }

      function playing(s){
        if(!first) document.getElementById("player").setAttribute("src",'https://embed.spotify.com/?uri=spotify%3Atrack%3A'+songs[s.id]["spotifyID"]+'&theme=white');
        e=Math.tanh(Math.pow(songs[s.id][kind]+1,3))*20+Math.pow((songs[s.id][kind]+1),3);
        s.graphics.clear().beginStroke(s.name).beginFill("#ffffff").drawCircle(0, 0, 16+e).endFill();
        s.scaleX*=1.8;
        s.scaleY*=1.8;
        if(previous){
          if(previous.id!=s.id){
            var color=stage.getChildAt(songs.length+2*previous.id).name;
            e=Math.tanh(Math.pow(songs[previous.id][kind]+1,3))*20+Math.pow((songs[previous.id][kind]+1),3);
            stage.getChildAt(songs.length+2*previous.id).graphics.beginStroke("#777777").beginFill(color).drawCircle(0,0,16+e).endFill();
            stage.getChildAt(songs.length+2*previous.id).scaleX/=1.8;
            stage.getChildAt(songs.length+2*previous.id).scaleY/=1.8;
          }
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

        function hoverbw(element) {
          element.setAttribute('src', 'icon/backwardhigh.png');
        }
        function unhoverbw(element) {
          element.setAttribute('src', 'icon/backward.png');
        }

        function hoverfw(element) {
          element.setAttribute('src', 'icon/forwardhigh.png');
        }
        function unhoverfw(element) {
          element.setAttribute('src', 'icon/forward.png');
        }

        function hoverdw(element) {
          if(element.getAttribute("src")!="icon/drawhigh.png") element.setAttribute('src', 'icon/drawhili.png');
        }
        function unhoverdw(element) {
          if(element.getAttribute("src")!="icon/drawhigh.png") element.setAttribute('src', 'icon/draw.png');
        }

        function hovertn(element) {
          element.setAttribute('src', 'icon/whitenhili.png');
        }
        function unhovertn(element) {
          element.setAttribute('src', 'icon/whiten.png');
        }

        function clickdowntn(element) {
          element.setAttribute('src', 'icon/whitenhigh.png');
        }
        function clickuptn(element) {
          element.setAttribute('src', 'icon/whitenhili.png');
        }

        function colorpopup() {
            var popup = document.getElementById("myPopup");
            popup.classList.toggle("show");
        }

        function overlay(){
        	el = document.getElementById("overlay");
          document.getElementById("list").innerHTML=note;
        	el.style.visibility = "visible";
          document.getElementById("time").style.visibility="visible";
          document.getElementById("drlist").blur();
          document.getElementById("time").focus();
          document.getElementById("overlay").addEventListener("click", function(event){
            if(event.target.id=="overlay"||event.target.id=="time"){
              document.getElementById("overlay").style.visibility="hidden";
              document.getElementById("time").style.visibility="hidden";
              note="Your playlist:<br>";
              document.getElementById("overlay").removeAllEventListeners("click");
            }
          });
          document.addEventListener("keypress", function(e){
            if(e.keyCode==13){
              document.getElementById("overlay").style.visibility="hidden";
              document.getElementById("time").style.visibility="hidden";
              note="Your playlist:<br>";
              document.getElementById("overlay").removeAllEventListeners("keypress");
            }
          });
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
      <div class="explorer">
        <a id="badge" class="navbar-brand" href="?word=">Music Visualisation with t-sne</a>
        <form id="search" action="" method="GET">
          <input id="genres" type="hidden" name="genres" value="blues" />
          <input id="colone" type="hidden" name="colone" value="tender" />
          <input id="coltwo" type="hidden" name="coltwo" value="hiphopurban" />
          <input id="colthree" type="hidden" name="colthree" value="joy" />
          <input id="searchForm" class="drop2" type="submit" value="Search"/>
          <div>
            <input class="drop0" id="autocomplete" type="text" title="type &quot;a&quot;" name="word" value=<?php echo "'".$_GET["word"]."'"?>/>
          </div>
          <p style="float:right; margin-top:15px; margin-right:10px; color:#777777"> Type artist or word: </p>
        </form>
      </div>
    </nav>
    <div id="results"></div

    <!-- Begin page content -->
    <div class="container">
      <canvas id="demoCanvas" width="4000" height="1560"> </canvas>
    </div>

    <div id="overlay">
      <span id="time">&times;</span>
      <div id="advise">
        <p id="list"></p>
      </div>
    </div>

    <footer class="footer">
      <div class="explorer">
        <div class="vertical">
          <iframe id="player" src="https://embed.spotify.com/?uri=spotify%3Atrack%3A<?php echo $json[sizeof($json)-1]['spotifyID'] ?>&theme=white" frameborder="0" allowtransparency="true"></iframe>
        </div>
        <input id="next" class="manager" type="image" src="icon/forward.png" onmouseover="hoverfw(this);" onmouseout="unhoverfw(this);" onmousedown="unhoverfw(this);" onmouseup="hoverfw(this);" width="20px" heigth="20px" onclick="next()">
        <input id="previous" class="manager" style="margin-left:30px" type="image" src="icon/backward.png" onmouseover="hoverbw(this);" onmouseout="unhoverbw(this);" onmousedown="unhoverbw(this);" onmouseup="hoverbw(this);" width="20px" heigth="20px" onclick="prec()">
        <input id="drlist" class="play" style="margin-left:20px" type="image" src="icon/draw.png" onmouseover="hoverdw(this);" onmouseout="unhoverdw(this);" width="30px" heigth="30px" onclick="playlist()" value="Draw a playlist">
        <input id="whlist" class="manager" type="image" src="icon/whiten.png" onmouseover="hovertn(this);" onmouseout="unhovertn(this);" onmousedown="clickdowntn(this);" onmouseup="clickuptn(this);" width="20px" heigth="20px" onclick="empty()" value="Clear playlist">
        <div style="min-width:560px">
          <div style="float:left; min-width: 200px; overflow: hidden; display: inline-block">
            <p style="float:left; margin-top:35px; color:#777777"> Feature size and color: </p>
            <select id="dgenre" class="drop1" onchange="init()">
              <option <?php if ($_GET['genres']=="joy")  echo 'selected="selected"'; ?> value"joy">Joy</option>
              <option <?php if ($_GET['genres']=="sad")  echo 'selected="selected"'; ?> value"sad">Sad</option>
              <option <?php if ($_GET['genres']=="angry")  echo 'selected="selected"'; ?> value"angry">Angry</option>
              <option <?php if ($_GET['genres']=="erotic")  echo 'selected="selected"'; ?> value"erotic">Erotic</option>
              <option <?php if ($_GET['genres']=="tender")  echo 'selected="selected"'; ?> value"tender">Tender</option>
              <option <?php if ($_GET['genres']=="fear")  echo 'selected="selected"'; ?> value"fear">Fear</option>
              <option <?php if ($_GET['genres']=="blues")  echo 'selected="selected"'; ?> value="blues">Blues</option>
              <option <?php if ($_GET['genres']=="country")  echo 'selected="selected"'; ?> value="country">Country</option>
              <option <?php if ($_GET['genres']=="easylistening")  echo 'selected="selected"'; ?> value="easylistening">Easy Listening</option>
              <option <?php if ($_GET['genres']=="electronica")  echo 'selected="selected"'; ?> value="electronica">Electronica</option>
              <option <?php if ($_GET['genres']=="folk")  echo 'selected="selected"'; ?> value="folk">Folk</option>
              <option <?php if ($_GET['genres']=="hiphopurban")  echo 'selected="selected"'; ?> value="hiphopurban">Hip Hop Urban</option>
              <option <?php if ($_GET['genres']=="jazz")  echo 'selected="selected"'; ?> value="jazz">Jazz</option>
              <option <?php if ($_GET['genres']=="latin")  echo 'selected="selected"'; ?> value="latin">Latin</option>
              <option <?php if ($_GET['genres']=="newage")  echo 'selected="selected"'; ?> value="newage">New Age</option>
              <option <?php if ($_GET['genres']=="pop")  echo 'selected="selected"'; ?> value="pop">Pop</option>
              <option <?php if ($_GET['genres']=="rnbsoul")  echo 'selected="selected"'; ?> value="rnbsoul">R'n'b Soul</option>
              <option <?php if ($_GET['genres']=="rock")  echo 'selected="selected"'; ?> value="rock">Rock</option>
              <option <?php if ($_GET['genres']=="gospel")  echo 'selected="selected"'; ?> value="gospel">Gospel</option>
              <option <?php if ($_GET['genres']=="reggae")  echo 'selected="selected"'; ?> value="reggae">Reggae</option>
              <option <?php if ($_GET['genres']=="world")  echo 'selected="selected"'; ?> value="world">World</option>
            </select>
          </div>
          <div style="float:left; min-width: 200px; display: inline-block">
            <select class="drop1" id="coloruno" style="color: cyan; text-shadow: 1px 0 2px #777777;" onchange="init()">
              <option <?php if ($_GET['colone']=="joy")  echo 'selected="selected"'; ?> value"joy">Joy</option>
              <option <?php if ($_GET['colone']=="sad")  echo 'selected="selected"'; ?> value"sad">Sad</option>
              <option <?php if ($_GET['colone']=="angry")  echo 'selected="selected"'; ?> value"angry">Angry</option>
              <option <?php if ($_GET['colone']=="erotic")  echo 'selected="selected"'; ?> value"erotic">Erotic</option>
              <option <?php if ($_GET['colone']=="tender")  echo 'selected="selected"'; ?> value"tender">Tender</option>
              <option <?php if ($_GET['colone']=="fear")  echo 'selected="selected"'; ?> value"fear">Fear</option>
              <option <?php if ($_GET['colone']=="blues")  echo 'selected="selected"'; ?> value="blues">Blues</option>
              <option <?php if ($_GET['colone']=="country")  echo 'selected="selected"'; ?> value="country">Country</option>
              <option <?php if ($_GET['colone']=="easylistening")  echo 'selected="selected"'; ?> value="easylistening">Easy Listening</option>
              <option <?php if ($_GET['colone']=="electronica")  echo 'selected="selected"'; ?> value="electronica">Electronica</option>
              <option <?php if ($_GET['colone']=="folk")  echo 'selected="selected"'; ?> value="folk">Folk</option>
              <option <?php if ($_GET['colone']=="hiphopurban")  echo 'selected="selected"'; ?> value="hiphopurban">Hip Hop Urban</option>
              <option <?php if ($_GET['colone']=="jazz")  echo 'selected="selected"'; ?> value="jazz">Jazz</option>
              <option <?php if ($_GET['colone']=="latin")  echo 'selected="selected"'; ?> value="latin">Latin</option>
              <option <?php if ($_GET['colone']=="newage")  echo 'selected="selected"'; ?> value="newage">New Age</option>
              <option <?php if ($_GET['colone']=="pop")  echo 'selected="selected"'; ?> value="pop">Pop</option>
              <option <?php if ($_GET['colone']=="rnbsoul")  echo 'selected="selected"'; ?> value="rnbsoul">R'n'b Soul</option>
              <option <?php if ($_GET['colone']=="rock")  echo 'selected="selected"'; ?> value="rock">Rock</option>
              <option <?php if ($_GET['colone']=="gospel")  echo 'selected="selected"'; ?> value="gospel">Gospel</option>
              <option <?php if ($_GET['colone']=="reggae")  echo 'selected="selected"'; ?> value="reggae">Reggae</option>
              <option <?php if ($_GET['colone']=="world")  echo 'selected="selected"'; ?> value="world">World</option>
            </select>
            <select class="drop1" id="colordue" style="color: magenta; text-shadow: 1px 0 2px #777777;" onchange="init()">
              <option <?php if ($_GET['coltwo']=="joy")  echo 'selected="selected"'; ?> value"joy">Joy</option>
              <option <?php if ($_GET['coltwo']=="sad")  echo 'selected="selected"'; ?> value"sad">Sad</option>
              <option <?php if ($_GET['coltwo']=="angry")  echo 'selected="selected"'; ?> value"angry">Angry</option>
              <option <?php if ($_GET['coltwo']=="erotic")  echo 'selected="selected"'; ?> value"erotic">Erotic</option>
              <option <?php if ($_GET['coltwo']=="tender")  echo 'selected="selected"'; ?> value"tender">Tender</option>
              <option <?php if ($_GET['coltwo']=="fear")  echo 'selected="selected"'; ?> value"fear">Fear</option>
              <option <?php if ($_GET['coltwo']=="blues")  echo 'selected="selected"'; ?> value="blues">Blues</option>
              <option <?php if ($_GET['coltwo']=="country")  echo 'selected="selected"'; ?> value="country">Country</option>
              <option <?php if ($_GET['coltwo']=="easylistening")  echo 'selected="selected"'; ?> value="easylistening">Easy Listening</option>
              <option <?php if ($_GET['coltwo']=="electronica")  echo 'selected="selected"'; ?> value="electronica">Electronica</option>
              <option <?php if ($_GET['coltwo']=="folk")  echo 'selected="selected"'; ?> value="folk">Folk</option>
              <option <?php if ($_GET['coltwo']=="hiphopurban")  echo 'selected="selected"'; ?> value="hiphopurban">Hip Hop Urban</option>
              <option <?php if ($_GET['coltwo']=="jazz")  echo 'selected="selected"'; ?> value="jazz">Jazz</option>
              <option <?php if ($_GET['coltwo']=="latin")  echo 'selected="selected"'; ?> value="latin">Latin</option>
              <option <?php if ($_GET['coltwo']=="newage")  echo 'selected="selected"'; ?> value="newage">New Age</option>
              <option <?php if ($_GET['coltwo']=="pop")  echo 'selected="selected"'; ?> value="pop">Pop</option>
              <option <?php if ($_GET['coltwo']=="rnbsoul")  echo 'selected="selected"'; ?> value="rnbsoul">R'n'b Soul</option>
              <option <?php if ($_GET['coltwo']=="rock")  echo 'selected="selected"'; ?> value="rock">Rock</option>
              <option <?php if ($_GET['coltwo']=="gospel")  echo 'selected="selected"'; ?> value="gospel">Gospel</option>
              <option <?php if ($_GET['coltwo']=="reggae")  echo 'selected="selected"'; ?> value="reggae">Reggae</option>
              <option <?php if ($_GET['coltwo']=="world")  echo 'selected="selected"'; ?> value="world">World</option>
            </select>
            <select class="drop1" id="colortre" style="color: yellow; text-shadow: 1px 0 2px #777777;" onchange="init()">
              <option <?php if ($_GET['colthree']=="joy")  echo 'selected="selected"'; ?> value"joy">Joy</option>
              <option <?php if ($_GET['colthree']=="sad")  echo 'selected="selected"'; ?> value"sad">Sad</option>
              <option <?php if ($_GET['colthree']=="angry")  echo 'selected="selected"'; ?> value"angry">Angry</option>
              <option <?php if ($_GET['colthree']=="erotic")  echo 'selected="selected"'; ?> value"erotic">Erotic</option>
              <option <?php if ($_GET['colthree']=="tender")  echo 'selected="selected"'; ?> value"tender">Tender</option>
              <option <?php if ($_GET['colthree']=="fear")  echo 'selected="selected"'; ?> value"fear">Fear</option>
              <option <?php if ($_GET['colthree']=="blues")  echo 'selected="selected"'; ?> value="blues">Blues</option>
              <option <?php if ($_GET['colthree']=="country")  echo 'selected="selected"'; ?> value="country">Country</option>
              <option <?php if ($_GET['colthree']=="easylistening")  echo 'selected="selected"'; ?> value="easylistening">Easy Listening</option>
              <option <?php if ($_GET['colthree']=="electronica")  echo 'selected="selected"'; ?> value="electronica">Electronica</option>
              <option <?php if ($_GET['colthree']=="folk")  echo 'selected="selected"'; ?> value="folk">Folk</option>
              <option <?php if ($_GET['colthree']=="hiphopurban")  echo 'selected="selected"'; ?> value="hiphopurban">Hip Hop Urban</option>
              <option <?php if ($_GET['colthree']=="jazz")  echo 'selected="selected"'; ?> value="jazz">Jazz</option>
              <option <?php if ($_GET['colthree']=="latin")  echo 'selected="selected"'; ?> value="latin">Latin</option>
              <option <?php if ($_GET['colthree']=="newage")  echo 'selected="selected"'; ?> value="newage">New Age</option>
              <option <?php if ($_GET['colthree']=="pop")  echo 'selected="selected"'; ?> value="pop">Pop</option>
              <option <?php if ($_GET['colthree']=="rnbsoul")  echo 'selected="selected"'; ?> value="rnbsoul">R'n'b Soul</option>
              <option <?php if ($_GET['colthree']=="rock")  echo 'selected="selected"'; ?> value="rock">Rock</option>
              <option <?php if ($_GET['colthree']=="gospel")  echo 'selected="selected"'; ?> value="gospel">Gospel</option>
              <option <?php if ($_GET['colthree']=="reggae")  echo 'selected="selected"'; ?> value="reggae">Reggae</option>
              <option <?php if ($_GET['colthree']=="world")  echo 'selected="selected"'; ?> value="world">World</option>
            </select>
            <div class="popup" onclick="colorpopup()" >
              <span class="popuptext" id="myPopup">Keep in mind how color mix!</span>
            </div>
            <img class="image" onclick="colorpopup()" src="icon/CMY_ideal_version.svg" alt="CMYK_image" style="width:50px;height:50px;">
          </div>
      </div>
    </footer>


    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->

    <script src="jquery-ui-1.12.1.custom/external/jquery/jquery.js"></script>
    <script src="jquery-ui-1.12.1.custom/jquery-ui.js"></script>
    <script>
      var availableTags = <?php echo  json_encode($fillings, JSON_PRETTY_PRINT)?>;

      $("#autocomplete").autocomplete({
        source: availableTags,
        appendTo: "#results",
        open: function(){
          var position = $("#results").position(),
          left = position.left, top = position.top;

          $("#results > ul").css({left: (left + 20) + "px",
                                  top: (top + 4) + "px" });
        },
        select: function(event, ui) {
       $("#searchField").val(ui.item.label);
       $("#searchForm").submit(); }
      });

      $(window).on('resize', function(){
        var win = $(this); //this = window
        if (win.width() < 700) { $(".ui-autocomplete").css("display","none") }
        if (win.width() > 700) { $(".ui-autocomplete").css("display","display") }
      });

    </script>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
    <script>window.jQuery || document.write('<script src="bootstrap-3.3.7/docs/assets/js/vendor/jquery.min.js"><\/script>')</script>
    <script src="bootstrap-3.3.7/docs/dist/js/bootstrap.min.js"></script>
    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <script src="bootstrap-3.3.7/docs/assets/js/ie10-viewport-bug-workaround.js"></script>
  </body>
</html>
