//
//  WebVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-06-14.
//
/*
 Display some HTML in a WKWebView.
 Good tutorial: https://www.amerhukic.com/determining-the-content-size-of-a-wkwebview
 */

import UIKit
import WebKit

//========================================================
class WebVC: AHXVC, WKUIDelegate {
    let html = """
    <HTML>
    <HEAD>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, shrink-to-fit=no">
    </HEAD>
    <STYLE>
    ol { padding-left: 1.8em; }
    </STYLE>
    
    <BODY>
    <ol>
    <li>Sit still for 5 seconds and then we keep adding text because I want to see the line break.</li>
    <li>Walk to the pylon.</li>
    <li>Come back.</li>
    </ol>
    </BODY>
    </HTML>
    """
    var webView:WKWebView!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        let webConfiguration = WKWebViewConfiguration()
        self.webView = WKWebView( frame: .zero, configuration: webConfiguration)
        self.webView.uiDelegate = self
    } // viewDidLoad()
    
    //-------------------------
    override func layout() {
        AHL.width( self.webView, self.view.frame.width / 2 )
        AHL.height( self.webView, self.view.frame.height / 2)
        AHL.subcenter( self.webView, self.view)
        AHL.submiddle( self.webView, self.view)
        AHL.border( self.webView)
    } // layout()
    
    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.webView.loadHTMLString( self.html, baseURL: nil)
    } // viewWillAppear()
    
} // class WebVC
