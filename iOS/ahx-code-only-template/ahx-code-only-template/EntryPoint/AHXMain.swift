//
//  AHXMain.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=====================
class AHXMain {
    static let shared = AHXMain()
    static var app = UIApplication.shared.delegate as! AppDelegate
    static var scene = SceneDelegate.shared!
    var navVC:UINavigationController!
    var VCs = [String:UIViewController]()
    
    // Entry Point to App, called from SceneDelegate
    //------------------------------------------------
    func main( navVC:UINavigationController) {
        // Some globally useful constants
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        
        self.navVC = navVC
        navVC.setNavigationBarHidden( true, animated:false)
        VCs = [
            "FirstVC": FirstVC(),
            "SecondVC": SecondVC()
            //"BottomNavVC": BottomNavVC(),
            //"ContainerVC": ContainerVC()
        ]
        // Force instantiation
        //let _ = VCs["BottomNavVC"]!.view
        
        pushVC( "FirstVC")
        //AHXMain.scene = VCs["FirstVC"]!.view.window!.windowScene!.delegate
    } // main()
    
    // Push a VC by name
    //-------------------------------
    func pushVC( _ vcName:String) {
        let vc = VCs[vcName]!
        let _ = vc.view // Make sure it's instantiated
        self.navVC.pushViewController( vc, animated: true)
    } // pushCV()

    // Pop top VC
    //----------------
    func popVC() {
        //let vc = VCs[vcName]!
        self.navVC.popViewController(animated: true)
    } // popCV()

    // Replace top VC
    //-------------------------------
    func topVC( _ vcName:String) {
        let vc = VCs[vcName]!
        self.navVC.popViewController( animated:true)
        self.navVC.pushViewController( vc, animated: true)
    } // topVC()
    
} // class AHXMain
