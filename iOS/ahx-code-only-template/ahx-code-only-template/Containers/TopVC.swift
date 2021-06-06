//
//  TopVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// Navigation bar at the top of the screen

import UIKit

//=================================
class TopVC: UIViewController {
    static var shared:TopVC!
    var btnBack:UIButton!
    var btnBurger:UIButton!
    var burgerMenu:AHXSlideMenu!
    var menuItems = ["One","Two"]
    var menuActions = [
        { ()->() in
            AHP.popup( title: "Menu", message: "One") },
        { ()->() in
            AHP.popup( title: "Menu", message: "Two") }
    ]
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        // Let taps through to children outside the frame.
        // In particular, the burger menu must be tappable.
        self.view = AHXPassThruView()
        
        TopVC.shared = self
        btnBack = UIButton( type: .system,
                            primaryAction: UIAction(title: "Back", handler: { _ in
                              AHXMain.shared.popVC()
                            }))
        self.view.addSubview( btnBack)
        
        btnBurger = UIButton( type: .custom,
                              primaryAction: UIAction() { _ in
                                self.burgerMenu.show()
                              })
        btnBurger.setImage( UIImage( named:"burger"), for: .normal)
        self.view.addSubview( btnBurger)
        burgerMenu = AHXSlideMenu( items: self.menuItems, actions: self.menuActions)
        self.view.addSubview( burgerMenu)
    } // viewDidLoad()
    
//    //------------------------------------------------------------------------------------
//    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
//        let touch = touches.first
//        var tt=42
//        //onTouch( touch!)
//    } // touchesBegan()

    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        layout()
    } // viewWillAppear()
    
    //--------------------------
    override func layout() {
        // Layout root frame
        var v = self.view!
        AHXLayout.width( v, AHC.w)
        AHXLayout.height( v, AHC.top_nav_height)
        AHXLayout.left( v, 0)
        AHXLayout.top( v, AHC.top)
        
        // Back button
        v = btnBack
        AHXLayout.width( v, v.intrinsicContentSize.width)
        AHXLayout.height( v, view.frame.height)
        AHXLayout.left( v, AHC.lmarg)
        AHXLayout.middle( v, view.frame.height / 2)
        //AHXLayout.border( v, .blue)
        
        // Burger Menu
        v = btnBurger
        AHXLayout.width( v, view.frame.width * 0.1)
        AHL.scaleHeight( v, likeImage: UIImage( named:"burger")!)
        AHXLayout.height( v, view.frame.height * 0.5)
        AHXLayout.right( v, view.frame.width - AHC.rmarg)
        AHXLayout.middle( v, view.frame.height / 2)

    } // layout()
} // class TopVC

