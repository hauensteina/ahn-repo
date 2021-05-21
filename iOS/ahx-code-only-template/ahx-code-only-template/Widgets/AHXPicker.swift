//
// AHXPicker.swift
//
// Created by Andreas Hauenstein on 2021-05-21
// Copyright © 2018 PROTXX, INC. All rights reserved.
//

// A utility class to add a pickerview to a textfield
/* Example usage:
 let myTextField = AHXPickerTf()
 let picker = AHXPicker( tf:myTextField,
                        choices: ["one","two"]) { (idx:Int) in  let choice = idx }
 */

import UIKit

//======================================================================
class AHXPicker: NSObject, UIPickerViewDelegate, UIPickerViewDataSource
{
    var tf = UITextField()
    var choices = [String]()
    var onCancel: ()->()
    var onDone: (_ idx:Int)->()
    let picker = UIPickerView()
    private var choiceIdx:Int = 0
    
    //-----------------------------------------------------------------------------------
    init( tf:UITextField, choices:[String], completion: @escaping (_ idx:Int)->() ) {
        self.tf = tf
        self.choices = choices
        self.onCancel = { () in } // Set this if you reallly need it
        self.onDone = completion
        super.init()
        picker.delegate = self
        tf.tintColor = .clear // no cursor
        tf.isUserInteractionEnabled = true
        
        let toolbar = UIToolbar(frame: CGRect(x: 0, y: 0, width: 100.0, height: 44.0)) // fix constraint warning
        toolbar.sizeToFit()
        let doneButton = UIBarButtonItem( title: "Done", style: UIBarButtonItem.Style.plain,
                                          target: self, action: #selector(done))
        let spaceButton = UIBarButtonItem( barButtonSystemItem: UIBarButtonItem.SystemItem.flexibleSpace,
                                           target: nil, action: nil)
        let cancelButton = UIBarButtonItem( title: "Cancel", style: UIBarButtonItem.Style.plain,
                                            target: self, action: #selector(cancel))
        toolbar.setItems( [cancelButton,spaceButton,doneButton], animated: false)
        tf.inputAccessoryView = toolbar
        tf.inputView = picker
    } // init()
    
    //---------------------
    @objc func cancel() {
        let vc = AHU.myVC( tf)
        vc.view.endEditing(true)
        onCancel()
    } // cancel()
    
    //-------------------
    @objc func done() {
        let vc = AHU.myVC( tf)
        vc.view.endEditing(true)
        tf.text = choices[self.choiceIdx]
        onDone( self.choiceIdx)
    } // done()

    // UIPickerView Delegation
    //============================
    func numberOfComponents( in pickerView: UIPickerView) -> Int { return 1 }
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return self.choices.count
    }
    func pickerView( _ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return choices[row]
    }
    func pickerView( _ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        self.choiceIdx = row
    }
} // class AHXPicker

// A subclass uf UITextfield with editing disabled
//===================================================
class AHXPickerTf: UITextField {
    open override func canPerformAction(_ action: Selector, withSender sender: Any?) -> Bool {
        return false
    }
} // AHXPickerTf()
