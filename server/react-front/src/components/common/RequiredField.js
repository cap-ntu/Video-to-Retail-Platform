import React from "react";
import * as PropTypes from "prop-types"
import assert from "assert";
import {emptyFunction} from "../../utils/utils";

class RequiredField extends React.PureComponent {

    state = {
        error: false
    };

    validate = () => {
        const {value: _value, initValue} = this.props;
        const {value} = React.Children.toArray(this.props.children)[0].props;
        const error = (_value !== undefined ? _value : value) === initValue;

        this.setState({error: error});

        return error;
    };

    handleChange = e => {
        const {bindingChange} = this.props;
        const {[bindingChange]: onChange = emptyFunction} = React.Children.toArray(this.props.children)[0].props;
        onChange(e, this.validate);
    };

    render() {
        const {children, bindingChange} = this.props;
        const {error} = this.state;

        const _children = React.Children.toArray(children);

        assert.strictEqual(_children.length, 1, `Expected exactly one child, but got ${_children.length}`);

        const {helperText} = _children[0].props;

        return (
            <React.Fragment>
                {
                    React.Children.map(children, element =>
                        React.cloneElement(element,
                            {
                                required: true,
                                error: error,
                                helperText: error ? "This field is required" : helperText,
                                [bindingChange]: this.handleChange,
                                onBlur: this.validate,
                            })
                    )
                }
            </React.Fragment>
        );
    }
}

RequiredField.defaultProps = {
    bindingChange: "onChange",
};

RequiredField.propTypes = {
    value: PropTypes.any,
    initValue: PropTypes.any.isRequired,
    children: PropTypes.node.isRequired,
    bindingChange: PropTypes.string,
};

export default RequiredField;
