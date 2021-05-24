//
//  AHXPopups.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-05-23.
//

import UIKit

class AHXPopups {
    // Display a popup
    //-----------------------------------------------------------------------------
    class func popup( title:String, message: String)->Void {
        DispatchQueue.main.async {
            if AHXMain.currentVC().presentedViewController != nil {
                AHXLog("popup(): already presenting")
                return
            }
            let alert = UIAlertController( title: title, message: message, preferredStyle: .alert)
            alert.addAction( UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))
            AHXMain.currentVC().present( alert, animated: true, completion: nil)
        }
    } // popup
    
    // Display error popup
    //---------------------------------------------
    class func errPopup( message: String)->Void {
        if AHXMain.currentVC().presentedViewController != nil {
            AHXLog("errPopup(): already presenting")
            return
        }
        DispatchQueue.main.async {
            AHXPopups.popup( title: "Error", message: message)
        }
    } // errPopup
    
    // Display a popup and do something on OK
    //--------------------------------------------------------
    class func actionPopup( title:String, message: String,
                            completion: @escaping ()->())
    {
        if AHXMain.currentVC().presentedViewController != nil {
            AHXLog("actionPopup(): already presenting")
            return
        }
        DispatchQueue.main.async {
            let alert = UIAlertController( title: title, message: message, preferredStyle: .alert)
            alert.addAction( UIAlertAction(title: "OK", style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion() }))
            AHXMain.currentVC().present( alert, animated: true, completion: nil)
        }
    } // actionPopup()
    
    // Display a popup with OK or Cancel
    //-----------------------------------------------------------------------------
    class func cancelPopup( title:String, message: String,
                            completion: @escaping (String)->())
    {
        if AHXMain.currentVC().presentedViewController != nil {
            AHXLog("cancelPopup(): already presenting")
            return
        }
        DispatchQueue.main.async {
            let alert = UIAlertController( title: title, message: message, preferredStyle: .alert)
            alert.addAction( UIAlertAction(title: "OK", style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion("ok") }))
            alert.addAction( UIAlertAction(title: "Cancel", style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion("cancel") }))
            AHXMain.currentVC().present( alert, animated: true, completion: nil)
        }
    } // cancelPopup()
    
    // Display a popup with a text input.
    //-----------------------------------------------------------------------------
    class func inputPopup( title:String, message: String,
                           completion: @escaping (String, String)->())
    {
        if AHXMain.currentVC().presentedViewController != nil {
            AHXLog("inputPopup(): already presenting")
            return
        }
        DispatchQueue.main.async {
            let alert = UIAlertController( title: title, message: message, preferredStyle: .alert)
            var txt = UITextField()
            alert.addTextField { (textField) in txt = textField }
            alert.addAction( UIAlertAction(title: "Cancel", style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion("", "cancel") }))
            alert.addAction( UIAlertAction(title: "OK", style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion( txt.text!, "ok") }))
            AHXMain.currentVC().present( alert, animated: true, completion: nil)
        }
        
    } // inputPopup()
    
    // Display a popup with two choices
    //------------------------------------------------------------------------------
    class func choicePopup( title:String, message: String, choice1: String,
                            choice2: String,
                            completion: @escaping (String)->())
    {
        if AHXMain.currentVC().presentedViewController != nil {
            AHXLog("choicePopup(): already presenting")
            return
        }
        DispatchQueue.main.async {
            let alert = UIAlertController( title: title, message: message, preferredStyle: .alert)
            alert.addAction( UIAlertAction(title: choice1, style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion("choice1") }))
            alert.addAction( UIAlertAction(title: choice2, style: UIAlertAction.Style.default,
                                           handler: { (alert: UIAlertAction!) in
                                            completion("choice2") }))
            AHXMain.currentVC().present( alert, animated: true, completion: nil)
        }
    } // cancelPopup()
    
} // class AHXPopups
