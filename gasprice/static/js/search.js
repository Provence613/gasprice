function searchInfo(){
var code = event.keyCode;
if(code == 13){
    var info=document.getElementById("input-info").value;
    //alert(info.length)
    if(info.length<8)
window.location.href="/datal?info="+info+"&type=block";
    else
window.location.href="/datal?info="+info+"&type=txn";
}

}

