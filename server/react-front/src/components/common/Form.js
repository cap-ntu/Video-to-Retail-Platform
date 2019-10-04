import React from 'react';

class Form extends React.PureComponent {
  handleSubmitWrapper = Object.freeze(handleSubmit => () => {
    const result = [];

    // call for validate, collect validation
    Object.values(this.child).forEach(v => result.push(!v.current.validate()));

    // check errors
    if (result.every(x => x)) handleSubmit();
  });

  constructor(props) {
    super(props);
    this.child = {};
    if (new.target === Form) {
      throw new TypeError('Cannot construct Abstract instances directly');
    }
  }
}

export default Form;
