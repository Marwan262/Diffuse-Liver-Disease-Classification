$(document).ready(function() {

    $(document).ready(function(){
        $("#myBtn").click(function(){
          $("#myModal").modal();
        });
      });
})

function generatePassword() {
    var string =(Math.random() + 1).toString(36).substring(2);
    document.getElementById("pass").value = string;
    document.getElementById("save").removeAttribute("hidden"); 
    document.getElementById("copy").removeAttribute("hidden"); 
}

function copyPassword() {
    var copyText = document.getElementById("pass");
    copyText.select();
    navigator.clipboard.writeText(copyText.value); 
}
