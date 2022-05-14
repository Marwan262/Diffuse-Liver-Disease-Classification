window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }

});

function validateName() {
    let name = document.forms["addForm"]["name"].value;
    if (/\d/.test(name)) {
        document.getElementById("nameError").hidden = false;
        document.getElementById("nameError").innerHTML = "Name cannot contain numbers.";
    }
    else 
        document.getElementById("nameError").hidden = true;
}

function validateNumber() {
    let number = document.forms["addForm"]["phoneno"].value;
    if (number.match(/[^$,.\d]/)) {
        document.getElementById("numberError").hidden = false;
        document.getElementById("numberError").innerHTML = "Name cannot contain numbers.";
    }
    else 
        document.getElementById("numberError").hidden = true;
}

function toggleEditReport() {
    x = document.getElementById('notes');
    if (x.style.display != 'none') {
    
        document.getElementById('notes').style.display = 'none';
        document.getElementById('notesE').style.display = 'block';
    
        document.getElementById('classification').style.display = 'none';
        document.getElementById('classificationE').style.display = 'block';

        document.getElementById('editSubmit').style.display = 'block';
    }
    else if (x.style.display == 'none') {

        document.getElementById('notes').style.display = 'block';
        document.getElementById('notesE').style.display = 'none';

        document.getElementById('classification').style.display = 'block';
        document.getElementById('classificationE').style.display = 'none';

        document.getElementById('editSubmit').style.display = 'none';
    } 
}

function toggleCheckbox() {
    x = document.getElementById('diagnosisCheck');
    if (x.checked == true) {
        $("#diagnosis").prop('readonly', false);
    }
    else {
        $("#diagnosis").prop('readonly', true);
    }
}

$( document ).ready(function() {
    var w = window.innerWidth;
   
    if(w > 767){
        $('#menu-jk').scrollToFixed();
    }else{
        $('#menu-jk').scrollToFixed();
    }
    
})




$(document).ready(function(){

    $(".filter-button").click(function(){
        var value = $(this).attr('data-filter');
        
        if(value == "all")
        {
            //$('.filter').removeClass('hidden');
            $('.filter').show('1000');
        }
        else
        {
//            $('.filter[filter-item="'+value+'"]').removeClass('hidden');
//            $(".filter").not('.filter[filter-item="'+value+'"]').addClass('hidden');
            $(".filter").not('.'+value).hide('3000');
            $('.filter').filter('.'+value).show('3000');
            
        }
    });
    
    if ($(".filter-button").removeClass("active")) {
$(this).removeClass("active");
}
$(this).addClass("active");

});

