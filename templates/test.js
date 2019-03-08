$(function() {
    var additemtoform = $('#additemform');
    var handleData = function(responseData) {
        if ($(responseData).find('#repitems').length) {
            var replacement = $('<div />').html(responseData).find('#repitems').html();
            $('#test').html(replacement);
        } else {
            location.href = "WebCatPageServer.exe?Cart";
        }
    };
    $("#additemform .addbtn").on("click", function(e) {
        e.preventDefault();
        $.ajax({
            type: 'POST',
            url: 'WebCatPageServer.exe',
            dataType: 'html',
            cache: false,
            data: additemtoform.serialize(),
            beforeSend: function() {},
            success: function(data) {
                handleData(data);
            },
            error: function() {
                alert('error');
            }
        });
        return false;
    });
});