// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Called when the user clicks on the browser action.
var title = document.title;
console.log(title);
chrome.extension.onMessage.addListener(
    function(request, sender, sendMessage) {
    	if (request.greeting == "hello")
            sendMessage(title);
        else
            sendMessage("FUCK OFF"); // snub them.
        }
);